//! Two-pass workgroup reduction (sum / max) — Step 3 of the GPU roadmap.
//!
//! Pass 1 dispatches `buckets` workgroups; each reduces a grid-strided slice
//! of the input into one partial value (`partials[workgroup_id]`).  Pass 2
//! dispatches a single workgroup over the partials buffer and writes the final
//! value.  Both passes run the same WGSL entry point (`sum_main` / `max_main`)
//! through two bind groups, so no second shader is needed.
//!
//! The workgroup-level combine uses `var<workgroup>` shared memory with
//! barrier-separated tree reduction — the same shared-memory pattern the tiled
//! GEMM uses — which is why this kernel is part of the tiling roadmap.

const std = @import("std");
const wgpu = @import("webgpu.zig");
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

const PipelineSlot = struct {
    shader_module: wgpu.WGPUShaderModule = null,
    pipeline_layout: wgpu.WGPUPipelineLayout = null,
    pipeline: wgpu.WGPUComputePipeline = null,

    fn deinit(self: *PipelineSlot) void {
        if (self.pipeline) |handle| wgpu.wgpuComputePipelineRelease(handle);
        self.pipeline = null;
        if (self.pipeline_layout) |handle| wgpu.wgpuPipelineLayoutRelease(handle);
        self.pipeline_layout = null;
        if (self.shader_module) |handle| wgpu.wgpuShaderModuleRelease(handle);
        self.shader_module = null;
    }
};

const Cache = struct {
    device: wgpu.WGPUDevice = null,
    n: usize = 0,
    buckets: usize = 0,

    bind_group_layout: wgpu.WGPUBindGroupLayout = null,
    pipelines: [2]PipelineSlot = .{ .{}, .{} },

    input: wgpu.WGPUBuffer = null,
    partials: wgpu.WGPUBuffer = null,
    final: wgpu.WGPUBuffer = null,
    staging: wgpu.WGPUBuffer = null,
    params_in: wgpu.WGPUBuffer = null,
    params_partials: wgpu.WGPUBuffer = null,
    bind_group_in: wgpu.WGPUBindGroup = null,
    bind_group_partials: wgpu.WGPUBindGroup = null,

    fn deinitBuffers(self: *Cache) void {
        if (self.bind_group_in) |handle| wgpu.wgpuBindGroupRelease(handle);
        self.bind_group_in = null;
        if (self.bind_group_partials) |handle| wgpu.wgpuBindGroupRelease(handle);
        self.bind_group_partials = null;
        for ([_]*wgpu.WGPUBuffer{
            &self.input,
            &self.partials,
            &self.final,
            &self.staging,
            &self.params_in,
            &self.params_partials,
        }) |slot| {
            if (slot.*) |handle| wgpu.wgpuBufferRelease(handle);
            slot.* = null;
        }
        self.n = 0;
        self.buckets = 0;
    }

    fn deinit(self: *Cache) void {
        self.deinitBuffers();
        for (&self.pipelines) |*slot| slot.deinit();
        if (self.bind_group_layout) |handle| wgpu.wgpuBindGroupLayoutRelease(handle);
        self.bind_group_layout = null;
        self.device = null;
    }
};

var global_cache: Cache = .{};
var cache_mutex: std.atomic.Mutex = .unlocked;

fn lockCache() void {
    while (!cache_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn unlockCache() void {
    cache_mutex.unlock();
}

pub fn resetCache() void {
    lockCache();
    defer unlockCache();
    global_cache.deinit();
}

fn opIndex(op: Op) usize {
    return switch (op) {
        .sum => 0,
        .max => 1,
    };
}

fn emptyStringView() wgpu.WGPUStringView {
    return .{ .data = null, .length = wgpu.WGPU_STRLEN };
}

fn stringView(comptime text: []const u8) wgpu.WGPUStringView {
    return .{ .data = text.ptr, .length = text.len };
}

fn writeBytes(ctx: *GpuContext, buffer: wgpu.WGPUBuffer, bytes: []const u8) void {
    wgpu.wgpuQueueWriteBuffer(
        ctx.queue,
        buffer,
        0,
        @as(?*const anyopaque, @ptrCast(bytes.ptr)),
        bytes.len,
    );
}

/// Number of pass-1 workgroups: one per 64 elements, capped at the device's
/// per-dimension dispatch limit.  Each workgroup then handles several
/// 64-element strides via the shader's grid-stride loop.
fn bucketCount(limits: context_mod.GpuLimits, n: usize) usize {
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

fn createStorageBuffer(ctx: *GpuContext, byte_size: usize, usage: wgpu.WGPUBufferUsage) !wgpu.WGPUBuffer {
    return ctx.createBuffer(&wgpu.WGPUBufferDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .usage = usage,
        .size = @intCast(byte_size),
        .mappedAtCreation = 0,
    });
}

fn ensurePipelines(ctx: *GpuContext, cache: *Cache) !void {
    if (cache.bind_group_layout == null) {
        var entries: [3]wgpu.WGPUBindGroupLayoutEntry = undefined;
        for (&entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupLayoutEntry);
        entries[0].binding = 0;
        entries[0].visibility = wgpu.WGPUShaderStage_Compute;
        entries[0].buffer.type = wgpu.WGPUBufferBindingType_ReadOnlyStorage;
        entries[1].binding = 1;
        entries[1].visibility = wgpu.WGPUShaderStage_Compute;
        entries[1].buffer.type = wgpu.WGPUBufferBindingType_Storage;
        entries[2].binding = 2;
        entries[2].visibility = wgpu.WGPUShaderStage_Compute;
        entries[2].buffer.type = wgpu.WGPUBufferBindingType_Uniform;

        cache.bind_group_layout = try ctx.createBindGroupLayout(&wgpu.WGPUBindGroupLayoutDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .entryCount = entries.len,
            .entries = entries[0..].ptr,
        });
    }

    inline for ([2][]const u8{ "sum_main", "max_main" }, 0..) |entry_point, index| {
        const slot = &cache.pipelines[index];
        if (slot.pipeline == null) {
            errdefer slot.deinit();
            var bind_group_layouts = [1]wgpu.WGPUBindGroupLayout{cache.bind_group_layout};
            slot.pipeline_layout = try ctx.createPipelineLayout(&wgpu.WGPUPipelineLayoutDescriptor{
                .nextInChain = null,
                .label = emptyStringView(),
                .bindGroupLayoutCount = bind_group_layouts.len,
                .bindGroupLayouts = bind_group_layouts[0..].ptr,
                .immediateSize = 0,
            });

            var shader_source = wgpu.WGPUShaderSourceWGSL{
                .chain = .{
                    .next = null,
                    .sType = wgpu.WGPUSType_ShaderSourceWGSL,
                },
                .code = .{ .data = reduce_shader.ptr, .length = reduce_shader.len },
            };
            slot.shader_module = try ctx.createShaderModule(&wgpu.WGPUShaderModuleDescriptor{
                .nextInChain = @ptrCast(&shader_source.chain),
                .label = emptyStringView(),
            });

            slot.pipeline = try ctx.createComputePipeline(&wgpu.WGPUComputePipelineDescriptor{
                .nextInChain = null,
                .label = emptyStringView(),
                .layout = slot.pipeline_layout,
                .compute = .{
                    .nextInChain = null,
                    .module = slot.shader_module,
                    .entryPoint = stringView(entry_point),
                    .constantCount = 0,
                    .constants = null,
                },
            });
        }
    }
}

fn ensureBuffers(ctx: *GpuContext, cache: *Cache, n: usize, buckets: usize) !void {
    if (cache.n == n and cache.buckets == buckets and cache.bind_group_in != null) return;

    cache.deinitBuffers();
    errdefer cache.deinitBuffers();

    const input_bytes = inputBytes(n) orelse return error.GpuError;
    const partial_bytes = std.math.mul(usize, buckets, @sizeOf(f32)) catch return error.GpuError;

    cache.input = try createStorageBuffer(
        ctx,
        input_bytes,
        wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.partials = try createStorageBuffer(
        ctx,
        partial_bytes,
        wgpu.WGPUBufferUsage_Storage,
    );
    cache.final = try createStorageBuffer(
        ctx,
        @sizeOf(f32),
        wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc,
    );
    cache.staging = try createStorageBuffer(
        ctx,
        @sizeOf(f32),
        wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.params_in = try createStorageBuffer(
        ctx,
        @sizeOf(Params),
        wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.params_partials = try createStorageBuffer(
        ctx,
        @sizeOf(Params),
        wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst,
    );

    const bindings = [2]struct {
        input: wgpu.WGPUBuffer,
        output: wgpu.WGPUBuffer,
        params: wgpu.WGPUBuffer,
    }{
        .{ .input = cache.input, .output = cache.partials, .params = cache.params_in },
        .{ .input = cache.partials, .output = cache.final, .params = cache.params_partials },
    };

    var bind_groups = [2]wgpu.WGPUBindGroup{ null, null };
    for (bindings, 0..) |binding_set, index| {
        var entries: [3]wgpu.WGPUBindGroupEntry = undefined;
        for (&entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupEntry);
        const buffers = [3]wgpu.WGPUBuffer{
            binding_set.input,
            binding_set.output,
            binding_set.params,
        };
        for (0..3) |binding| {
            entries[binding].binding = @intCast(binding);
            entries[binding].buffer = buffers[binding];
            entries[binding].offset = 0;
            entries[binding].size = wgpu.WGPU_WHOLE_SIZE;
        }
        bind_groups[index] = try ctx.createBindGroup(&wgpu.WGPUBindGroupDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .layout = cache.bind_group_layout,
            .entryCount = entries.len,
            .entries = entries[0..].ptr,
        });
    }
    cache.bind_group_in = bind_groups[0];
    cache.bind_group_partials = bind_groups[1];

    cache.n = n;
    cache.buckets = buckets;
}

fn runImpl(
    ctx: *GpuContext,
    cache: *Cache,
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

    if (cache.device != ctx.device) {
        cache.deinit();
        cache.device = ctx.device;
    }
    try ensurePipelines(ctx, cache);
    try ensureBuffers(ctx, cache, n, buckets);

    writeBytes(ctx, cache.input, std.mem.sliceAsBytes(input));
    var params_in = Params{ .n = @intCast(n), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    writeBytes(ctx, cache.params_in, std.mem.asBytes(&params_in));
    var params_partials = Params{ .n = @intCast(buckets), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    writeBytes(ctx, cache.params_partials, std.mem.asBytes(&params_partials));

    const index = opIndex(op);
    ctx.beginErrorScope();

    const encoder = wgpu.wgpuDeviceCreateCommandEncoder(ctx.device, null) orelse {
        ctx.discardErrorScope();
        return error.GpuError;
    };
    for (0..repetitions) |_| {
        const pass = wgpu.wgpuCommandEncoderBeginComputePass(encoder, null) orelse {
            wgpu.wgpuCommandEncoderRelease(encoder);
            ctx.discardErrorScope();
            return error.GpuError;
        };
        wgpu.wgpuComputePassEncoderSetPipeline(pass, cache.pipelines[index].pipeline);
        wgpu.wgpuComputePassEncoderSetBindGroup(pass, 0, cache.bind_group_in, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast(buckets), 1, 1);
        // Pass 2 collapses the per-workgroup partials with a single workgroup.
        wgpu.wgpuComputePassEncoderSetBindGroup(pass, 0, cache.bind_group_partials, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        wgpu.wgpuComputePassEncoderEnd(pass);
        wgpu.wgpuComputePassEncoderRelease(pass);
    }
    wgpu.wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        cache.final,
        0,
        cache.staging,
        0,
        @sizeOf(f32),
    );
    const command_buffer = wgpu.wgpuCommandEncoderFinish(encoder, null) orelse {
        wgpu.wgpuCommandEncoderRelease(encoder);
        ctx.discardErrorScope();
        return error.GpuError;
    };
    wgpu.wgpuCommandEncoderRelease(encoder);

    var commands = [1]wgpu.WGPUCommandBuffer{command_buffer};
    wgpu.wgpuQueueSubmit(ctx.queue, 1, commands[0..].ptr);
    ctx.endErrorScope() catch |err| {
        wgpu.wgpuCommandBufferRelease(command_buffer);
        return err;
    };
    wgpu.wgpuCommandBufferRelease(command_buffer);

    try ctx.readBuffer(cache.staging, std.mem.asBytes(out));
}

/// Reduce `input` into `out` with the process-local GPU context and cache.
pub fn runWithContext(ctx: *GpuContext, op: Op, out: *f32, input: []const f32) !void {
    lockCache();
    defer unlockCache();
    return runImpl(ctx, &global_cache, op, out, input, 1);
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
    lockCache();
    defer unlockCache();
    return runImpl(ctx, &global_cache, op, out, input, repetitions);
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

/// 8-lane CPU reference.  The lane-wise partial sums use a different
/// association order than the scalar reference (and than the GPU), so callers
/// compare with a tolerance instead of bit equality.
pub fn referenceSumSimd(input: []const f32) f32 {
    const V = @Vector(8, f32);
    var lanes: V = @splat(0);
    var index: usize = 0;
    while (index + 8 <= input.len) : (index += 8) {
        const values: V = @as(*align(1) const V, @ptrCast(input.ptr + index)).*;
        lanes += values;
    }
    var total: f32 = @reduce(.Add, lanes);
    while (index < input.len) : (index += 1) total += input[index];
    return total;
}

pub fn referenceMaxSimd(input: []const f32) f32 {
    const V = @Vector(8, f32);
    var lanes: V = @splat(-std.math.inf(f32));
    var index: usize = 0;
    while (index + 8 <= input.len) : (index += 8) {
        const values: V = @as(*align(1) const V, @ptrCast(input.ptr + index)).*;
        lanes = @max(lanes, values);
    }
    var total: f32 = @reduce(.Max, lanes);
    while (index < input.len) : (index += 1) total = @max(total, input[index]);
    return total;
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
