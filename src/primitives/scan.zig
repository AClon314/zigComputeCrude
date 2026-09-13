//! Exclusive prefix sum (scan) over `u32` — the base primitive for
//! compaction and spatial indexing (docs/node-system-migration.md, M3).
//!
//! Three dispatches are recorded into the caller's `Chain` (one submit, one
//! readback policy as usual).  The `Scanner` owns the kernels, the block
//! partials buffer and the params buffer; a `Binding` owns the bind groups for
//! one (input, output) buffer pair, so steady-state callers create it once.
//!
//! Layout contract (matches scan.wgsl): 64 elements per workgroup, partials has
//! `ceil(count/64)` entries.  `max_blocks` is fixed at init so buffers never
//! reallocate under a live `Binding`.

const std = @import("std");
const runtime = @import("../runtime.zig");
const wgpu = @import("../gpu/webgpu.zig");
const context_mod = @import("../gpu/context.zig");

const scan_shader = @embedFile("../gpu/shaders/scan.wgsl");

pub const workgroup_size: usize = 64;

pub const Params = extern struct {
    count: u32,
    block_count: u32,
    pad0: u32,
    pad1: u32,
};

pub fn blockCount(count: usize) usize {
    return (count + workgroup_size - 1) / workgroup_size;
}

/// CPU reference: `output[i] = sum(input[0..i])`.
pub fn referenceExclusiveScan(input: []const u32, output: []u32) void {
    std.debug.assert(input.len == output.len);
    var running: u32 = 0;
    for (input, 0..) |value, index| {
        output[index] = running;
        running += value;
    }
}

pub const Scanner = struct {
    ctx: *runtime.Device,
    block_scan_kernel: runtime.Kernel,
    block_global_kernel: runtime.Kernel,
    apply_kernel: runtime.Kernel,
    partials: runtime.Buffer,
    params: runtime.Buffer,
    max_blocks: usize,

    pub fn init(ctx: *runtime.Device, max_blocks: usize) !Scanner {
        if (max_blocks == 0) return error.GpuError;
        const bindings = [_]runtime.Binding{
            .{ .kind = .storage, .access = .read }, // input
            .{ .kind = .storage, .access = .write }, // output
            .{ .kind = .storage, .access = .write }, // partials
            .{ .kind = .uniform, .access = .read }, // params
        };

        var self = Scanner{
            .ctx = ctx,
            .block_scan_kernel = try runtime.Kernel.init(ctx, scan_shader, "block_scan", &bindings, workgroup_size),
            .block_global_kernel = try runtime.Kernel.init(ctx, scan_shader, "block_scan_global", &bindings, workgroup_size),
            .apply_kernel = try runtime.Kernel.init(ctx, scan_shader, "scan_apply", &bindings, workgroup_size),
            .partials = undefined,
            .params = undefined,
            .max_blocks = max_blocks,
        };
        errdefer {
            self.block_scan_kernel.deinit();
            self.block_global_kernel.deinit();
            self.apply_kernel.deinit();
        }
        self.partials = try runtime.Buffer.init(
            ctx,
            max_blocks * @sizeOf(u32),
            runtime.buffer.storage_rw,
        );
        self.params = try runtime.Buffer.init(ctx, 16, runtime.buffer.uniform);
        return self;
    }

    pub fn deinit(self: *Scanner) void {
        self.partials.deinit(self.ctx);
        self.params.deinit(self.ctx);
        self.block_scan_kernel.deinit();
        self.block_global_kernel.deinit();
        self.apply_kernel.deinit();
    }

    /// Bind groups for one (input, output) pair.  Create once, reuse for every
    /// scan of that pair; release with `deinit`.
    pub fn bind(self: *Scanner, input: *const runtime.Buffer, output: *const runtime.Buffer) !Binding {
        return .{
            .block_scan = try self.block_scan_kernel.createBindGroup(&.{ input, output, &self.partials, &self.params }),
            .block_global = try self.block_global_kernel.createBindGroup(&.{ input, output, &self.partials, &self.params }),
            .apply = try self.apply_kernel.createBindGroup(&.{ input, output, &self.partials, &self.params }),
            .input = input,
            .output = output,
        };
    }

    pub fn release(self: *Scanner, binding: *Binding) void {
        _ = self;
        runtime.releaseBindGroup(binding.block_scan);
        runtime.releaseBindGroup(binding.block_global);
        runtime.releaseBindGroup(binding.apply);
        binding.* = undefined;
    }

    /// Record `output = exclusive_scan(input, count)` into `chain`.
    ///
    /// Params are uploaded immediately (outside the chain), so a Scanner
    /// instance records at most one logical scan per chain.
    pub fn run(
        self: *Scanner,
        chain: *runtime.Chain,
        binding: *const Binding,
        input: *runtime.Buffer,
        output: *runtime.Buffer,
        count: usize,
    ) !void {
        if (count == 0) return;
        if (count > input.byte_size / @sizeOf(u32) or count > output.byte_size / @sizeOf(u32)) {
            return error.GpuError;
        }
        const blocks = blockCount(count);
        if (blocks > self.max_blocks) return error.GpuError;

        var params = Params{
            .count = @intCast(count),
            .block_count = @intCast(blocks),
            .pad0 = 0,
            .pad1 = 0,
        };
        self.params.markHostDirty();
        try self.params.toDevice(self.ctx, std.mem.asBytes(&params));

        const grid = try self.block_scan_kernel.gridLinear(count);
        const single = context_mod.WorkgroupGrid{ .x = 1, .y = 1 };
        const buffers = [_]*runtime.Buffer{ input, output, &self.partials, &self.params };

        try chain.dispatch(&self.block_scan_kernel, binding.block_scan, &buffers, grid);
        try chain.dispatch(&self.block_global_kernel, binding.block_global, &buffers, single);
        try chain.dispatch(&self.apply_kernel, binding.apply, &buffers, grid);
    }
};

pub const Binding = struct {
    block_scan: wgpu.WGPUBindGroup,
    block_global: wgpu.WGPUBindGroup,
    apply: wgpu.WGPUBindGroup,
    input: *const runtime.Buffer,
    output: *const runtime.Buffer,
};

// ---- tests ----

fn scanWithGpu(allocator: std.mem.Allocator, values: []const u32) ![]u32 {
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const count = values.len;
    const blocks = blockCount(count);
    var scanner = try Scanner.init(ctx, @max(blocks, 1));
    defer scanner.deinit();

    var input = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_r);
    defer input.deinit(ctx);
    var output = try runtime.Buffer.init(ctx, count * @sizeOf(u32), runtime.buffer.storage_rw);
    defer output.deinit(ctx);
    try input.toDevice(ctx, std.mem.sliceAsBytes(values));

    var binding = try scanner.bind(&input, &output);
    defer scanner.release(&binding);

    const result = try allocator.alloc(u32, count);
    errdefer allocator.free(result);

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    try scanner.run(&chain, &binding, &input, &output, count);
    try chain.download(&output, std.mem.sliceAsBytes(result));
    try chain.submit();
    return result;
}

test "scan matches the cpu reference" {
    const gpa = std.testing.allocator;
    const sizes = [_]usize{ 1, 2, 63, 64, 65, 127, 128, 129, 1000, 4097 };

    for (sizes) |count| {
        const input = try gpa.alloc(u32, count);
        defer gpa.free(input);
        const expected = try gpa.alloc(u32, count);
        defer gpa.free(expected);
        var prng = std.Random.DefaultPrng.init(count);
        const random = prng.random();
        for (input) |*value| value.* = random.uintLessThan(u32, 1000);
        referenceExclusiveScan(input, expected);

        const actual = try scanWithGpu(gpa, input);
        defer gpa.free(actual);
        try std.testing.expectEqualSlices(u32, expected, actual);
    }
}
