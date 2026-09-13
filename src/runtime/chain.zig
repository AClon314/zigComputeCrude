//! M0 runtime: record many dispatches into one command buffer and submit once.
//!
//! A `Chain` is the "resident + chained" half of the M0 goal: the caller keeps
//! its buffers on the device, records a sequence of dispatches (possibly with
//! different kernels), reads back only what it needs, and pays the host round
//! trip exactly once.
//!
//! Ablation contract (see docs/node-system-migration.md and the `--kernel chain`
//! CLI): the only differences between "per-call", "per-submit" and "chained"
//! modes are how many submits/readbacks happen, never what is computed.

const std = @import("std");
const wgpu = @import("../gpu/webgpu.zig");
const context_mod = @import("../gpu/context.zig");
const GpuContext = context_mod.GpuContext;
const Buffer = @import("buffer.zig").Buffer;
const Kernel = @import("kernel.zig").Kernel;

pub const max_downloads = 4;

const PendingDownload = struct {
    buffer: *Buffer,
    out: []u8,
    /// Prefix length to copy; may be shorter than the buffer (capacity-sized
    /// buffers, e.g. an index built for `max_points` but queried with fewer).
    byte_size: usize,
};

pub const Chain = struct {
    ctx: *GpuContext,
    encoder: wgpu.WGPUCommandEncoder = null,
    pass: wgpu.WGPUComputePassEncoder = null,
    dispatches: usize = 0,
    downloads: [max_downloads]PendingDownload = undefined,
    download_count: usize = 0,
    submitted: bool = false,

    pub fn begin(ctx: *GpuContext) !Chain {
        ctx.beginErrorScope();
        const encoder = wgpu.wgpuDeviceCreateCommandEncoder(ctx.device, null) orelse {
            ctx.discardErrorScope();
            return error.GpuError;
        };
        return .{ .ctx = ctx, .encoder = encoder };
    }

    /// Record one dispatch.  `buffers` must match `kernel.bindings` in order;
    /// writable buffers are marked device-newest so the next `toDevice` for
    /// them can be skipped.
    pub fn dispatch(
        self: *Chain,
        kernel: *const Kernel,
        bind_group: wgpu.WGPUBindGroup,
        buffers: []const *Buffer,
        grid: context_mod.WorkgroupGrid,
    ) !void {
        if (self.submitted) return error.GpuError;
        if (buffers.len != kernel.bindings.len) return error.GpuError;

        if (self.pass == null) {
            self.pass = wgpu.wgpuCommandEncoderBeginComputePass(self.encoder, null) orelse
                return error.GpuError;
        }
        wgpu.wgpuComputePassEncoderSetPipeline(self.pass, kernel.slot.pipeline);
        wgpu.wgpuComputePassEncoderSetBindGroup(self.pass, 0, bind_group, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(self.pass, grid.x, grid.y, 1);
        self.dispatches += 1;

        for (buffers, 0..) |buffer, index| {
            if (kernel.bindings[index].access == .write) buffer.markDeviceWritten();
        }
    }

    /// Record one dispatch whose workgroup count is read from `indirect`
    /// (3 x u32: x, y, z, the `WGPU_INDIRECT` layout).  Useful for dynamic
    /// work sizes such as compaction outputs; the caller is responsible for
    /// writing the count before this dispatch in the same chain (or earlier).
    pub fn dispatchIndirect(
        self: *Chain,
        kernel: *const Kernel,
        bind_group: wgpu.WGPUBindGroup,
        buffers: []const *Buffer,
        indirect: *const Buffer,
        indirect_offset: u64,
    ) !void {
        if (self.submitted) return error.GpuError;
        if (buffers.len != kernel.bindings.len) return error.GpuError;
        if (indirect_offset + 12 > indirect.byte_size) return error.GpuError;

        if (self.pass == null) {
            self.pass = wgpu.wgpuCommandEncoderBeginComputePass(self.encoder, null) orelse
                return error.GpuError;
        }
        wgpu.wgpuComputePassEncoderSetPipeline(self.pass, kernel.slot.pipeline);
        wgpu.wgpuComputePassEncoderSetBindGroup(self.pass, 0, bind_group, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            self.pass,
            indirect.handle,
            indirect_offset,
        );
        self.dispatches += 1;

        for (buffers, 0..) |buffer, index| {
            if (kernel.bindings[index].access == .write) buffer.markDeviceWritten();
        }
    }

    /// Register a readback (a prefix of the buffer is allowed).  The copy is
    /// recorded after all dispatches (i.e. it reads the final state of the
    /// chain) and `out` must stay alive until `submit` returns.
    pub fn download(self: *Chain, buffer: *Buffer, out: []u8) !void {
        if (self.submitted) return error.GpuError;
        if (out.len == 0 or out.len > buffer.byte_size) return error.GpuError;
        if (self.download_count == self.downloads.len) return error.GpuError;
        self.downloads[self.download_count] = .{
            .buffer = buffer,
            .out = out,
            .byte_size = out.len,
        };
        self.download_count += 1;
    }

    /// End the pass, copy downloads to staging, submit once, then map/copy the
    /// requested outputs.  The encoder and all staging mappings are released
    /// before returning.
    pub fn submit(self: *Chain) !void {
        if (self.submitted) return error.GpuError;
        self.submitted = true;

        if (self.pass) |pass| {
            wgpu.wgpuComputePassEncoderEnd(pass);
            wgpu.wgpuComputePassEncoderRelease(pass);
            self.pass = null;
        }

        for (self.downloads[0..self.download_count]) |item| {
            try item.buffer.ensureStaging(self.ctx);
            wgpu.wgpuCommandEncoderCopyBufferToBuffer(
                self.encoder,
                item.buffer.handle,
                0,
                item.buffer.staging,
                0,
                @intCast(item.byte_size),
            );
        }

        const encoder = self.encoder orelse return error.GpuError;
        self.encoder = null;
        try self.ctx.submitRecorded(encoder);

        for (self.downloads[0..self.download_count]) |item| {
            try self.ctx.readBuffer(item.buffer.staging, item.out);
            item.buffer.sync = .synced;
        }
    }

    /// Release resources if `submit` was never called (error paths).
    pub fn deinit(self: *Chain) void {
        if (self.pass) |pass| {
            wgpu.wgpuComputePassEncoderEnd(pass);
            wgpu.wgpuComputePassEncoderRelease(pass);
            self.pass = null;
        }
        if (self.encoder) |encoder| {
            wgpu.wgpuCommandEncoderRelease(encoder);
            self.encoder = null;
            self.ctx.discardErrorScope();
        }
    }
};

// ---- tests ----

const indirect_probe_shader = @embedFile("../gpu/shaders/indirect_probe.wgsl");
const indirect_fill_shader = @embedFile("../gpu/shaders/indirect_fill.wgsl");

test "chain dispatchIndirect honours a gpu-written workgroup count" {
    const gpa = std.testing.allocator;
    const ctx = @import("../runtime.zig").open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const CountParams = extern struct { limit: u32, pad0: u32, pad1: u32, pad2: u32 };
    const FillParams = extern struct { value: u32, pad0: u32, pad1: u32, pad2: u32 };
    const value: u32 = 0x5A5A_1234;

    var set_count = try Kernel.init(ctx, indirect_probe_shader, "set_count", &.{
        .{ .kind = .storage, .access = .write },
        .{ .kind = .uniform, .access = .read },
    }, 1);
    defer set_count.deinit();
    var fill = try Kernel.init(ctx, indirect_fill_shader, "fill", &.{
        .{ .kind = .storage, .access = .write },
        .{ .kind = .uniform, .access = .read },
    }, 64);
    defer fill.deinit();

    var control = try Buffer.init(ctx, 3 * @sizeOf(u32), @import("buffer.zig").indirect);
    defer control.deinit(ctx);
    const data_elements: usize = 4096;
    var data = try Buffer.init(ctx, data_elements * @sizeOf(u32), @import("buffer.zig").storage_rw);
    defer data.deinit(ctx);
    var count_params = try Buffer.init(ctx, 16, @import("buffer.zig").uniform);
    defer count_params.deinit(ctx);
    var fill_params = try Buffer.init(ctx, 16, @import("buffer.zig").uniform);
    defer fill_params.deinit(ctx);

    const count_values = CountParams{ .limit = 3, .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    try count_params.toDevice(ctx, std.mem.asBytes(&count_values));
    const fill_values = FillParams{ .value = value, .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    try fill_params.toDevice(ctx, std.mem.asBytes(&fill_values));

    const zeros = try gpa.alloc(u32, data_elements);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    try control.toDevice(ctx, std.mem.asBytes(&[3]u32{ 0, 1, 1 }));
    try data.toDevice(ctx, std.mem.sliceAsBytes(zeros));

    const bg_count = try set_count.createBindGroup(&.{ &control, &count_params });
    defer @import("../runtime.zig").releaseBindGroup(bg_count);
    const bg_fill = try fill.createBindGroup(&.{ &data, &fill_params });
    defer @import("../runtime.zig").releaseBindGroup(bg_fill);

    const result = try gpa.alloc(u32, data_elements);
    defer gpa.free(result);

    var chain = try Chain.begin(ctx);
    defer chain.deinit();
    try chain.dispatch(&set_count, bg_count, &.{ &control, &count_params }, .{ .x = 1, .y = 1 });
    try chain.dispatchIndirect(&fill, bg_fill, &.{ &data, &fill_params }, &control, 0);
    try chain.download(&data, std.mem.sliceAsBytes(result));
    try chain.submit();

    // limit=3 workgroups x 64 invocations = the first 192 elements are stamped.
    for (result, 0..) |got, index| {
        const expected: u32 = if (index < 3 * 64) value else 0;
        try std.testing.expectEqual(expected, got);
    }
}
