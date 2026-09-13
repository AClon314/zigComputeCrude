//! Two-pass workgroup reduction (sum / max) — Step 3 of the GPU roadmap,
//! migrated to the M0 runtime (M1).
//!
//! Pass 1 dispatches `buckets` workgroups; each reduces a grid-strided slice
//! of the input into one partial value (`partials[workgroup_id]`).  Pass 2
//! dispatches a single workgroup over the partials buffer and writes the final
//! value.  Both passes run the same WGSL entry point (`sum_main` / `max_main`)
//! through two bind groups, so no second shader is needed.
//!
//! `Kernels` is the composable half: compiled sum/max pipelines plus a `bind`
//! helper returning the two bind groups (input->partials, partials->final) for
//! caller-owned runtime buffers.  The slice-based `reduce`/`reduceBatched`
//! entry points keep the historical per-call behaviour.

const std = @import("std");
const runtime = @import("../runtime.zig");
const context_mod = @import("context.zig");
const GpuContext = context_mod.GpuContext;

const reduce_shader = @embedFile("shaders/reduce.wgsl");

const workgroup_size: usize = 64;

pub const Op = enum {
    sum,
    max,

    pub fn name(self: Op) []const u8 {
        return switch (self) {
            .sum => "sum",
            .max => "max",
        };
    }
};

const Params = extern struct {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

const kernel_bindings = [_]runtime.Binding{
    .{ .kind = .storage, .access = .read }, // input
    .{ .kind = .storage, .access = .write }, // output (partials or final)
    .{ .kind = .uniform, .access = .read }, // params
};

fn opIndex(op: Op) usize {
    return switch (op) {
        .sum => 0,
        .max => 1,
    };
}

/// Compiled pipelines for one device; buffer-agnostic and cheap to share.
pub const Kernels = struct {
    ctx: *runtime.Device,
    sum: runtime.Kernel,
    max: runtime.Kernel,

    pub fn init(ctx: *runtime.Device) !Kernels {
        var self = Kernels{
            .ctx = ctx,
            .sum = try runtime.Kernel.init(ctx, reduce_shader, "sum_main", &kernel_bindings, workgroup_size),
            .max = undefined,
        };
        errdefer self.sum.deinit();
        self.max = try runtime.Kernel.init(ctx, reduce_shader, "max_main", &kernel_bindings, workgroup_size);
        return self;
    }

    pub fn deinit(self: *Kernels) void {
        self.max.deinit();
        self.sum.deinit();
    }

    pub fn kernel(self: *const Kernels, op: Op) *const runtime.Kernel {
        return if (op == .sum) &self.sum else &self.max;
    }

    /// Both stages' bind groups for one op: `first` = (input, partials,
    /// params_in), `second` = (partials, final, params_partials).  Created
    /// per op so no group-equivalence assumptions are needed.
    pub fn bindStage(
        self: *const Kernels,
        op: Op,
        input: *const runtime.Buffer,
        output: *const runtime.Buffer,
        params: *const runtime.Buffer,
    ) !?*anyopaque {
        return self.kernel(op).createBindGroup(&.{ input, output, params });
    }
};

/// All bind groups a full two-pass reduction needs.
pub const Binding = struct {
    first: [2]?*anyopaque, // input -> partials
    second: [2]?*anyopaque, // partials -> final
};

/// Bind both stages for every op over the same buffers.
pub fn bindAll(
    kernels: *const Kernels,
    input: *const runtime.Buffer,
    partials: *const runtime.Buffer,
    final: *const runtime.Buffer,
    params_in: *const runtime.Buffer,
    params_partials: *const runtime.Buffer,
) !Binding {
    var binding = Binding{
        .first = .{ null, null },
        .second = .{ null, null },
    };
    for ([_]Op{ .sum, .max }) |op| {
        const index = opIndex(op);
        binding.first[index] = try kernels.bindStage(op, input, partials, params_in);
        binding.second[index] = try kernels.bindStage(op, partials, final, params_partials);
    }
    return binding;
}

/// Release a `Binding` produced by `bindAll`.
pub fn releaseBinding(binding: *Binding) void {
    for (&binding.first) |*handle| {
        if (handle.*) |value| runtime.releaseBindGroup(value);
        handle.* = null;
    }
    for (&binding.second) |*handle| {
        if (handle.*) |value| runtime.releaseBindGroup(value);
        handle.* = null;
    }
}

const ReduceCache = struct {
    ctx: *runtime.Device,
    n: usize,
    buckets: usize,
    input: runtime.Buffer,
    partials: runtime.Buffer,
    final: runtime.Buffer,
    params_in: runtime.Buffer,
    params_partials: runtime.Buffer,
    binding: Binding,

    fn init(ctx: *runtime.Device, n: usize, buckets: usize) !ReduceCache {
        const input_bytes = inputBytes(n) orelse return error.GpuError;
        const partial_bytes = std.math.mul(usize, buckets, @sizeOf(f32)) catch return error.GpuError;

        var self = ReduceCache{
            .ctx = ctx,
            .n = n,
            .buckets = buckets,
            .input = undefined,
            .partials = undefined,
            .final = undefined,
            .params_in = undefined,
            .params_partials = undefined,
            .binding = .{ .first = .{ null, null }, .second = .{ null, null } },
        };
        self.input = try runtime.Buffer.init(ctx, input_bytes, runtime.buffer.storage_r);
        errdefer self.input.deinit(ctx);
        self.partials = try runtime.Buffer.init(ctx, partial_bytes, runtime.buffer.storage_rw);
        errdefer self.partials.deinit(ctx);
        self.final = try runtime.Buffer.init(ctx, @sizeOf(f32), runtime.buffer.storage_rw);
        errdefer self.final.deinit(ctx);
        self.params_in = try runtime.Buffer.init(ctx, @sizeOf(Params), runtime.buffer.uniform);
        errdefer self.params_in.deinit(ctx);
        self.params_partials = try runtime.Buffer.init(ctx, @sizeOf(Params), runtime.buffer.uniform);
        errdefer self.params_partials.deinit(ctx);
        return self;
    }

    fn deinit(self: *ReduceCache) void {
        releaseBinding(&self.binding);
        self.params_partials.deinit(self.ctx);
        self.params_in.deinit(self.ctx);
        self.final.deinit(self.ctx);
        self.partials.deinit(self.ctx);
        self.input.deinit(self.ctx);
    }
};

var kernels_cache: ?Kernels = null;
var shape_cache: ?ReduceCache = null;
var cache_mutex: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!cache_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn unlock() void {
    cache_mutex.unlock();
}

pub fn resetCache() void {
    lock();
    defer unlock();
    if (shape_cache) |*cache| cache.deinit();
    shape_cache = null;
    if (kernels_cache) |*kernels| kernels.deinit();
    kernels_cache = null;
}

/// Number of pass-1 workgroups: one per 64 elements, capped at the device's
/// per-dimension dispatch limit.  Each workgroup then handles several
/// 64-element strides via the shader's grid-stride loop.
pub fn bucketCount(limits: context_mod.GpuLimits, n: usize) usize {
    const raw = (n + workgroup_size - 1) / workgroup_size;
    const max_dim: usize = limits.maxComputeWorkgroupsPerDimension;
    return @min(raw, max_dim);
}

fn inputBytes(n: usize) ?usize {
    return std.math.mul(usize, n, @sizeOf(f32)) catch null;
}

/// True when the input, partials, params and 1D dispatch all fit the device.
pub fn canRun(limits: context_mod.GpuLimits, n: usize) bool {
    if (n == 0 or n > std.math.maxInt(u32)) return false;
    const bytes = inputBytes(n) orelse return false;
    const byte_count: u64 = @intCast(bytes);
    if (byte_count > limits.maxStorageBufferBindingSize or byte_count > limits.maxBufferSize) {
        return false;
    }
    const buckets = bucketCount(limits, n);
    return buckets > 0 and buckets <= limits.maxComputeWorkgroupsPerDimension;
}

fn ensureKernels(ctx: *runtime.Device) !void {
    if (kernels_cache) |*kernels| {
        if (kernels.ctx.device == ctx.device) return;
        kernels.deinit();
        kernels_cache = null;
        if (shape_cache) |*cache| cache.deinit();
        shape_cache = null;
    }
    kernels_cache = try Kernels.init(ctx);
}

fn ensureShape(ctx: *runtime.Device, n: usize, buckets: usize) !*ReduceCache {
    if (shape_cache) |*cache| {
        if (cache.ctx.device == ctx.device and cache.n == n and cache.buckets == buckets) {
            return cache;
        }
        cache.deinit();
        shape_cache = null;
    }

    var cache = try ReduceCache.init(ctx, n, buckets);
    errdefer cache.deinit();
    cache.binding = try bindAll(
        &kernels_cache.?,
        &cache.input,
        &cache.partials,
        &cache.final,
        &cache.params_in,
        &cache.params_partials,
    );
    shape_cache = cache;
    return &shape_cache.?;
}

fn runImpl(
    ctx: *GpuContext,
    op: Op,
    out: *f32,
    input: []const f32,
    repetitions: usize,
) !void {
    const n = input.len;
    if (repetitions == 0) return error.GpuError;
    if (!canRun(ctx.limits, n)) return error.GpuError;
    const buckets = bucketCount(ctx.limits, n);
    if (buckets == 0) return error.GpuError;

    lock();
    defer unlock();

    try ensureKernels(ctx);
    const cache = try ensureShape(ctx, n, buckets);

    cache.input.markHostDirty();
    try cache.input.toDevice(ctx, std.mem.sliceAsBytes(input));
    var params_in = Params{ .n = @intCast(n), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    cache.params_in.markHostDirty();
    try cache.params_in.toDevice(ctx, std.mem.asBytes(&params_in));
    var params_partials = Params{ .n = @intCast(buckets), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    cache.params_partials.markHostDirty();
    try cache.params_partials.toDevice(ctx, std.mem.asBytes(&params_partials));

    const index = opIndex(op);
    const kernel = kernels_cache.?.kernel(op);
    const buffers_first = [3]*runtime.Buffer{ &cache.input, &cache.partials, &cache.params_in };
    const buffers_second = [3]*runtime.Buffer{ &cache.partials, &cache.final, &cache.params_partials };
    const grid_first = try kernel.gridLinear(buckets);
    const grid_second = context_mod.WorkgroupGrid{ .x = 1, .y = 1 };

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    for (0..repetitions) |_| {
        try chain.dispatch(kernel, cache.binding.first[index].?, &buffers_first, grid_first);
        try chain.dispatch(kernel, cache.binding.second[index].?, &buffers_second, grid_second);
    }
    try chain.download(&cache.final, std.mem.asBytes(out));
    try chain.submit();
}

/// Reduce `input` into `out` with the process-local GPU context and cache.
pub fn runWithContext(ctx: *GpuContext, op: Op, out: *f32, input: []const f32) !void {
    return runImpl(ctx, op, out, input, 1);
}

/// Steady-state variant: upload once, run `repetitions` real two-pass
/// reductions, read the 4-byte result back once.
pub fn runBatchedWithContext(
    ctx: *GpuContext,
    op: Op,
    out: *f32,
    input: []const f32,
    repetitions: usize,
) !void {
    return runImpl(ctx, op, out, input, repetitions);
}

pub fn reduce(op: Op, out: *f32, input: []const f32) !void {
    context_mod.clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (context_mod.lastFallbackReason() == null) {
            context_mod.recordFallback(@errorName(err));
        }
        return err;
    };
    runWithContext(ctx, op, out, input) catch |err| {
        context_mod.recordFallback(@errorName(err));
        return err;
    };
}

pub fn reduceBatched(op: Op, out: *f32, input: []const f32, repetitions: usize) !void {
    context_mod.clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (context_mod.lastFallbackReason() == null) {
            context_mod.recordFallback(@errorName(err));
        }
        return err;
    };
    runBatchedWithContext(ctx, op, out, input, repetitions) catch |err| {
        context_mod.recordFallback(@errorName(err));
        return err;
    };
}

// ---- CPU references ----

pub fn referenceSum(input: []const f32) f32 {
    var acc: f32 = 0;
    for (input) |value| acc += value;
    return acc;
}

pub fn referenceMax(input: []const f32) f32 {
    var acc: f32 = -std.math.inf(f32);
    for (input) |value| acc = @max(acc, value);
    return acc;
}

/// Target-width CPU reference (vectorWidth() lanes).  The lane-wise partial
/// sums use a different association order than the scalar reference (and than
/// the GPU), so callers compare with a tolerance instead of bit equality.
pub fn referenceSumSimd(input: []const f32) f32 {
    const V = @Vector(vectorWidth(), f32);
    const width = vectorWidth();
    var lanes: V = @splat(0);
    var index: usize = 0;
    while (index + width <= input.len) : (index += width) {
        const values: V = @as(*align(1) const V, @ptrCast(input.ptr + index)).*;
        lanes += values;
    }
    var total: f32 = @reduce(.Add, lanes);
    while (index < input.len) : (index += 1) total += input[index];
    return total;
}

pub fn referenceMaxSimd(input: []const f32) f32 {
    const V = @Vector(vectorWidth(), f32);
    const width = vectorWidth();
    var lanes: V = @splat(-std.math.inf(f32));
    var index: usize = 0;
    while (index + width <= input.len) : (index += width) {
        const values: V = @as(*align(1) const V, @ptrCast(input.ptr + index)).*;
        lanes = @max(lanes, values);
    }
    var total: f32 = @reduce(.Max, lanes);
    while (index < input.len) : (index += 1) total = @max(total, input[index]);
    return total;
}

/// Target-dependent vector width (AVX2 = 8 f32, AVX-512 = 16, NEON = 4, …).
fn vectorWidth() comptime_int {
    return std.simd.suggestVectorLength(f32) orelse 4;
}

// ---- tests ----

fn fillDeterministic(data: []f32, seed: usize) void {
    for (data, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt((index * 13 + seed * 7) % 29)) * 0.5 - 4.0;
    }
}

test "reduce cpu references agree with the naive definition" {
    const gpa = std.testing.allocator;
    const n = 4099; // deliberately not a workgroup multiple
    const input = try gpa.alloc(f32, n);
    defer gpa.free(input);
    fillDeterministic(input, 1);

    var sum: f32 = 0;
    var max: f32 = -std.math.inf(f32);
    for (input) |value| {
        sum += value;
        max = @max(max, value);
    }
    try std.testing.expectEqual(sum, referenceSum(input));
    try std.testing.expectEqual(max, referenceMax(input));
}

test "reduce cpu scalar and simd references agree on edge sizes" {
    const gpa = std.testing.allocator;
    const sizes = [_]usize{ 1, 7, 8, 63, 64, 65, 1023, 4099 };
    for (sizes) |n| {
        const input = try gpa.alloc(f32, n);
        defer gpa.free(input);
        fillDeterministic(input, n);
        const sum_tolerance = @max(1.0, @abs(referenceSum(input))) * 1e-5;
        try std.testing.expect(@abs(referenceSum(input) - referenceSumSimd(input)) <= sum_tolerance);
        try std.testing.expectEqual(referenceMax(input), referenceMaxSimd(input));
    }
}

test "reduce gpu sum and max match the cpu references" {
    const gpa = std.testing.allocator;

    var context = GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();
    defer resetCache();

    const sizes = [_]usize{ 1, 63, 64, 65, 1023, 4099, 1 << 16 };
    for (sizes) |n| {
        const input = try gpa.alloc(f32, n);
        defer gpa.free(input);
        fillDeterministic(input, n);

        var gpu_sum: f32 = 0;
        try runWithContext(&context, .sum, &gpu_sum, input);
        const cpu_sum = referenceSum(input);
        const sum_tolerance = @max(1.0, @abs(cpu_sum)) * 1e-4;
        try std.testing.expect(@abs(cpu_sum - gpu_sum) <= sum_tolerance);

        var gpu_max: f32 = 0;
        try runWithContext(&context, .max, &gpu_max, input);
        try std.testing.expectEqual(referenceMax(input), gpu_max);
    }
}

test "reduce gpu sum is exact for exactly-representable inputs" {
    const gpa = std.testing.allocator;

    var context = GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();
    defer resetCache();

    // 1<<20 ones: every partial sum and every tree level stays below 2^24, so
    // the GPU result must be bit-exact, not just close.
    const n: usize = 1 << 20;
    const input = try gpa.alloc(f32, n);
    defer gpa.free(input);
    @memset(input, 1.0);

    var gpu_sum: f32 = 0;
    try runWithContext(&context, .sum, &gpu_sum, input);
    try std.testing.expectEqual(@as(f32, @floatFromInt(n)), gpu_sum);
}

test "reduce canRun honours storage and dispatch limits" {
    const limits = context_mod.GpuLimits{
        .maxComputeWorkgroupsPerDimension = 4,
        .maxStorageBufferBindingSize = 1024,
        .maxBufferSize = 2048,
    };

    // 256 f32 = 1 KiB, exactly the storage limit; 4 buckets fit the dimension.
    try std.testing.expect(canRun(limits, 256));
    try std.testing.expect(!canRun(limits, 257));
    try std.testing.expect(!canRun(limits, 0));

    // Large n is allowed as long as the bucket count is capped, but the input
    // must still fit the storage limit.
    var generous = limits;
    generous.maxStorageBufferBindingSize = 1 << 30;
    generous.maxBufferSize = 1 << 30;
    try std.testing.expect(canRun(generous, 1 << 20));
    try std.testing.expectEqual(@as(usize, 4), bucketCount(generous, 1 << 20));
}
