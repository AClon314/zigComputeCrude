//! Stream compaction (M3): keep the elements whose flag is non-zero, packed in
//! input order, and publish the result length.
//!
//! Three dispatches into the caller's `Chain`, built on the scan primitive:
//!
//!   scan flags -> offsets      (computeAccel.primitives.Scanner)
//!   scatter  output[offsets[i]] = values[i]  where flags[i] != 0
//!   total    count[0..3] = compacted length, 1, 1
//!
//! `count` is `Storage | CopyDst | CopySrc | Indirect`, so it can be passed
//! straight to `Chain.dispatchIndirect` for a follow-up pass over the compacted
//! prefix (this is what makes dynamic work sizes possible).
//!
//! Elements are 4-byte words: u32 or f32 payloads both work (bit-exact moves).

const std = @import("std");
const runtime = @import("../runtime.zig");
const wgpu = @import("../gpu/webgpu.zig");
const primitives = @import("../primitives.zig");

const scatter_shader = @embedFile("../gpu/shaders/compact_scatter.wgsl");
const total_shader = @embedFile("../gpu/shaders/compact_total.wgsl");

const Params = extern struct {
    count: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

/// CPU reference: compacts in place and returns the new length.
pub fn referenceCompact(values: []const u32, flags: []const u32, output: []u32) usize {
    std.debug.assert(values.len == flags.len and output.len >= values.len);
    var out: usize = 0;
    for (values, flags) |value, flag| {
        if (flag != 0) {
            output[out] = value;
            out += 1;
        }
    }
    return out;
}

pub const Compactor = struct {
    ctx: *runtime.Device,
    max_count: usize,

    offsets: runtime.Buffer,
    params: runtime.Buffer,
    /// [count, 1, 1], usable as a `dispatchIndirect` argument.
    count: runtime.Buffer,

    scatter_kernel: runtime.Kernel,
    total_kernel: runtime.Kernel,
    scanner: primitives.Scanner,

    pub fn init(ctx: *runtime.Device, max_count: usize) !Compactor {
        if (max_count == 0) return error.GpuError;

        var self = Compactor{
            .ctx = ctx,
            .max_count = max_count,
            .offsets = try runtime.Buffer.init(ctx, max_count * @sizeOf(u32), runtime.buffer.storage_rw),
            .params = undefined,
            .count = undefined,
            .scatter_kernel = undefined,
            .total_kernel = undefined,
            .scanner = undefined,
        };
        errdefer self.offsets.deinit(ctx);

        self.params = try runtime.Buffer.init(ctx, @sizeOf(Params), runtime.buffer.uniform);
        self.count = try runtime.Buffer.init(
            ctx,
            3 * @sizeOf(u32),
            runtime.buffer.storage_rw | wgpu.WGPUBufferUsage_Indirect,
        );
        self.scatter_kernel = try runtime.Kernel.init(ctx, scatter_shader, "compact_scatter", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);
        self.total_kernel = try runtime.Kernel.init(ctx, total_shader, "compact_total", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 1);
        self.scanner = try primitives.Scanner.init(ctx, primitives.scanBlockCount(max_count));
        return self;
    }

    pub fn deinit(self: *Compactor) void {
        self.scanner.deinit();
        self.total_kernel.deinit();
        self.scatter_kernel.deinit();
        self.count.deinit(self.ctx);
        self.params.deinit(self.ctx);
        self.offsets.deinit(self.ctx);
    }

    /// Bind groups for one buffer set: flags/values are inputs, output receives
    /// the packed prefix.  Create once and reuse across chains.
    pub fn bind(
        self: *Compactor,
        values: *const runtime.Buffer,
        flags: *const runtime.Buffer,
        output: *const runtime.Buffer,
    ) !Binding {
        return .{
            .scan = try self.scanner.bind(flags, &self.offsets),
            .scatter = try self.scatter_kernel.createBindGroup(&.{
                values,
                flags,
                &self.offsets,
                output,
                &self.params,
            }),
            .total = try self.total_kernel.createBindGroup(&.{
                flags,
                &self.offsets,
                &self.count,
                &self.params,
            }),
        };
    }

    pub fn release(self: *Compactor, binding: *Binding) void {
        self.scanner.release(&binding.scan);
        runtime.releaseBindGroup(binding.scatter);
        runtime.releaseBindGroup(binding.total);
        binding.* = undefined;
    }

    /// Record `output = compact(values, flags)` and publish the length in
    /// `self.count`.  All three dispatches share one chain; no readback is
    /// implied (the caller decides when to download).
    pub fn run(
        self: *Compactor,
        chain: *runtime.Chain,
        binding: *const Binding,
        values: *runtime.Buffer,
        flags: *runtime.Buffer,
        output: *runtime.Buffer,
        count: usize,
    ) !void {
        if (count == 0 or count > self.max_count) return error.GpuError;
        if (count > values.byte_size / @sizeOf(u32) or
            count > flags.byte_size / @sizeOf(u32) or
            count > output.byte_size / @sizeOf(u32))
        {
            return error.GpuError;
        }

        var params = Params{ .count = @intCast(count) };
        self.params.markHostDirty();
        try self.params.toDevice(self.ctx, std.mem.asBytes(&params));

        try self.scanner.run(chain, &binding.scan, flags, &self.offsets, count);

        const grid = try self.scatter_kernel.gridLinear(count);
        try chain.dispatch(&self.scatter_kernel, binding.scatter, &.{
            values,
            flags,
            &self.offsets,
            output,
            &self.params,
        }, grid);
        try chain.dispatch(&self.total_kernel, binding.total, &.{
            flags,
            &self.offsets,
            &self.count,
            &self.params,
        }, .{ .x = 1, .y = 1 });
    }
};

pub const Binding = struct {
    scan: primitives.ScanBinding,
    scatter: wgpu.WGPUBindGroup,
    total: wgpu.WGPUBindGroup,
};

// ---- tests ----

test "compaction matches the cpu reference and publishes an indirect count" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const count: usize = 5000;
    const values = try gpa.alloc(u32, count);
    defer gpa.free(values);
    const flags = try gpa.alloc(u32, count);
    defer gpa.free(flags);
    const expected = try gpa.alloc(u32, count);
    defer gpa.free(expected);
    var prng = std.Random.DefaultPrng.init(77);
    const random = prng.random();
    for (values) |*value| value.* = random.int(u32);
    for (flags, 0..) |*flag, index| flag.* = if (index % 3 == 0) 1 else 0;
    const expected_len = referenceCompact(values, flags, expected);

    var compactor = try Compactor.init(ctx, count);
    defer compactor.deinit();

    var values_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_r);
    defer values_buf.deinit(ctx);
    var flags_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_r);
    defer flags_buf.deinit(ctx);
    var output_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_rw);
    defer output_buf.deinit(ctx);
    try values_buf.toDevice(ctx, std.mem.sliceAsBytes(values));
    try flags_buf.toDevice(ctx, std.mem.sliceAsBytes(flags));

    var binding = try compactor.bind(&values_buf, &flags_buf, &output_buf);
    defer compactor.release(&binding);

    const actual = try gpa.alloc(u32, count);
    defer gpa.free(actual);
    const count_out = try gpa.alloc(u32, 3);
    defer gpa.free(count_out);

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    try compactor.run(&chain, &binding, &values_buf, &flags_buf, &output_buf, count);
    try chain.download(&output_buf, std.mem.sliceAsBytes(actual));
    try chain.download(&compactor.count, std.mem.sliceAsBytes(count_out));
    try chain.submit();

    try std.testing.expectEqual(@as(u32, @intCast(expected_len)), count_out[0]);
    try std.testing.expectEqual(@as(u32, 1), count_out[1]);
    try std.testing.expectEqual(@as(u32, 1), count_out[2]);
    try std.testing.expectEqualSlices(u32, expected[0..expected_len], actual[0..expected_len]);
}

test "compaction with no flags set yields an empty result" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const count: usize = 129; // crosses a scan block boundary
    const values = try gpa.alloc(u32, count);
    defer gpa.free(values);
    const flags = try gpa.alloc(u32, count);
    defer gpa.free(flags);
    @memset(values, 7);
    @memset(flags, 0);

    var compactor = try Compactor.init(ctx, count);
    defer compactor.deinit();
    var values_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_r);
    defer values_buf.deinit(ctx);
    var flags_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_r);
    defer flags_buf.deinit(ctx);
    var output_buf = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_rw);
    defer output_buf.deinit(ctx);
    try values_buf.toDevice(ctx, std.mem.sliceAsBytes(values));
    try flags_buf.toDevice(ctx, std.mem.sliceAsBytes(flags));

    var binding = try compactor.bind(&values_buf, &flags_buf, &output_buf);
    defer compactor.release(&binding);

    const count_out = try gpa.alloc(u32, 3);
    defer gpa.free(count_out);

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    try compactor.run(&chain, &binding, &values_buf, &flags_buf, &output_buf, count);
    try chain.download(&compactor.count, std.mem.sliceAsBytes(count_out));
    try chain.submit();

    try std.testing.expectEqual(@as(u32, 0), count_out[0]);
}
