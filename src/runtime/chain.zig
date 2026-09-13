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

    /// Register a readback.  The copy is recorded after all dispatches (i.e.
    /// it reads the final state of the chain) and `out` must stay alive until
    /// `submit` returns.
    pub fn download(self: *Chain, buffer: *Buffer, out: []u8) !void {
        if (self.submitted) return error.GpuError;
        if (out.len != buffer.byte_size) return error.GpuError;
        if (self.download_count == self.downloads.len) return error.GpuError;
        self.downloads[self.download_count] = .{ .buffer = buffer, .out = out };
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
                @intCast(item.buffer.byte_size),
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
