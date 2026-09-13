//! GEMM (A[m x k] * B[k x n] = C[m x n]) — Step 3 of the GPU roadmap.
//!
//! Two variants share one bind-group layout and one set of buffers:
//!
//!   * `simple` — one invocation per output element, global memory only.  This
//!     is the correctness baseline (and deliberately the slow one).
//!   * `tiled`  — a 16x16 output tile per 64-invocation workgroup with the A/B
//!     blocks staged in workgroup memory, so each input element is read from
//!     global memory once per tile row/column instead of once per output.
//!
//! The GPU path allocates once per (device, shape) and reuses the pipeline,
//! buffers and bind group across calls, matching the existing add/saxpy cache
//! strategy.  The public `gemm`/`gemmBatched` helpers use the process-local
//! context from `context.zig`; tests and callers that own a `GpuContext` use
//! the `*WithContext` entry points.

const std = @import("std");
const wgpu = @import("webgpu.zig");
const context_mod = @import("context.zig");
const GpuContext = context_mod.GpuContext;

const simple_shader = @embedFile("shaders/gemm_simple.wgsl");
const tiled_shader = @embedFile("shaders/gemm_tiled.wgsl");

pub const Variant = enum {
    simple,
    tiled,

    pub fn name(self: Variant) []const u8 {
        return switch (self) {
            .simple => "simple",
            .tiled => "tiled",
        };
    }
};

/// Workgroup tile edge of gemm_tiled.wgsl; keep the two in sync.
const tile_size: usize = 16;

const Params = extern struct {
    m: u32,
    k: u32,
    n: u32,
    pad: u32,
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
    m: usize = 0,
    k: usize = 0,
    n: usize = 0,

    bind_group_layout: wgpu.WGPUBindGroupLayout = null,
    pipelines: [2]PipelineSlot = .{ .{}, .{} },

    a: wgpu.WGPUBuffer = null,
    b: wgpu.WGPUBuffer = null,
    c: wgpu.WGPUBuffer = null,
    params: wgpu.WGPUBuffer = null,
    staging: wgpu.WGPUBuffer = null,
    bind_group: wgpu.WGPUBindGroup = null,

    fn deinitBuffers(self: *Cache) void {
        // Bind groups reference the buffers, so they go first.
        if (self.bind_group) |handle| wgpu.wgpuBindGroupRelease(handle);
        self.bind_group = null;
        for ([_]*wgpu.WGPUBuffer{ &self.a, &self.b, &self.c, &self.params, &self.staging }) |slot| {
            if (slot.*) |handle| wgpu.wgpuBufferRelease(handle);
            slot.* = null;
        }
        self.m = 0;
        self.k = 0;
        self.n = 0;
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

/// Drop every cached GPU resource.  Tests use this to release buffers/pipelines
/// while their own context is still alive; ordinary callers rely on the cache
/// being rebuilt when the device or shape changes.
pub fn resetCache() void {
    lockCache();
    defer unlockCache();
    global_cache.deinit();
}

fn variantIndex(variant: Variant) usize {
    return switch (variant) {
        .simple => 0,
        .tiled => 1,
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

const ValidShape = struct {
    m: u32,
    k: u32,
    n: u32,
    a_bytes: usize,
    b_bytes: usize,
    c_bytes: usize,
};

/// Validate dimensions once so every later index/product is known to fit both
/// the WGSL u32 index arithmetic and the host address space.
fn validateShape(m: usize, k: usize, n: usize) ?ValidShape {
    if (m == 0 or k == 0 or n == 0) return null;
    if (m > std.math.maxInt(u32) or k > std.math.maxInt(u32) or n > std.math.maxInt(u32)) {
        return null;
    }

    const u32_max: u64 = std.math.maxInt(u32);
    const a_elements = @as(u64, m) * k;
    const b_elements = @as(u64, k) * n;
    const c_elements = @as(u64, m) * n;
    if (a_elements > u32_max or b_elements > u32_max or c_elements > u32_max) {
        return null;
    }

    const a_bytes = std.math.mul(usize, @intCast(a_elements), @sizeOf(f32)) catch return null;
    const b_bytes = std.math.mul(usize, @intCast(b_elements), @sizeOf(f32)) catch return null;
    const c_bytes = std.math.mul(usize, @intCast(c_elements), @sizeOf(f32)) catch return null;
    return .{
        .m = @intCast(m),
        .k = @intCast(k),
        .n = @intCast(n),
        .a_bytes = a_bytes,
        .b_bytes = b_bytes,
        .c_bytes = c_bytes,
    };
}

/// True when the device can create every buffer and dispatch the grid this
/// (shape, variant) needs.  Selection logic can ask before committing.
pub fn canRun(limits: context_mod.GpuLimits, m: usize, k: usize, n: usize, variant: Variant) bool {
    const shape = validateShape(m, k, n) orelse return false;
    // The staging buffer is the same size as C.
    inline for (.{ shape.a_bytes, shape.b_bytes, shape.c_bytes }) |byte_size| {
        const bytes: u64 = @intCast(byte_size);
        if (bytes > limits.maxStorageBufferBindingSize or bytes > limits.maxBufferSize) return false;
    }

    switch (variant) {
        .simple => {
            const groups = (@as(usize, shape.m) * shape.n + 63) / 64;
            return limits.workgroupGrid(groups) != null;
        },
        .tiled => {
            const gx = (@as(usize, shape.n) + tile_size - 1) / tile_size;
            const gy = (@as(usize, shape.m) + tile_size - 1) / tile_size;
            const max_dim: u64 = limits.maxComputeWorkgroupsPerDimension;
            return gx <= max_dim and gy <= max_dim;
        },
    }
}

fn dispatchGridFor(
    variant: Variant,
    limits: context_mod.GpuLimits,
    shape: ValidShape,
) !context_mod.WorkgroupGrid {
    switch (variant) {
        .simple => {
            const groups = (@as(usize, shape.m) * shape.n + 63) / 64;
            return limits.workgroupGrid(groups) orelse error.GpuError;
        },
        .tiled => {
            const gx = (@as(usize, shape.n) + tile_size - 1) / tile_size;
            const gy = (@as(usize, shape.m) + tile_size - 1) / tile_size;
            const max_dim: u64 = limits.maxComputeWorkgroupsPerDimension;
            if (gx > max_dim or gy > max_dim) return error.GpuError;
            return .{ .x = @intCast(gx), .y = @intCast(gy) };
        },
    }
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
        var entries: [4]wgpu.WGPUBindGroupLayoutEntry = undefined;
        for (&entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupLayoutEntry);
        for (0..3) |binding| {
            entries[binding].binding = @intCast(binding);
            entries[binding].visibility = wgpu.WGPUShaderStage_Compute;
            entries[binding].buffer.type = if (binding == 2)
                wgpu.WGPUBufferBindingType_Storage
            else
                wgpu.WGPUBufferBindingType_ReadOnlyStorage;
        }
        entries[3].binding = 3;
        entries[3].visibility = wgpu.WGPUShaderStage_Compute;
        entries[3].buffer.type = wgpu.WGPUBufferBindingType_Uniform;

        cache.bind_group_layout = try ctx.createBindGroupLayout(&wgpu.WGPUBindGroupLayoutDescriptor{
            .nextInChain = null,
            .label = emptyStringView(),
            .entryCount = entries.len,
            .entries = entries[0..].ptr,
        });
    }

    const shader_code = [2][]const u8{ simple_shader, tiled_shader };
    for (&cache.pipelines, 0..) |*slot, index| {
        if (slot.pipeline != null) continue;

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
            .code = .{ .data = shader_code[index].ptr, .length = shader_code[index].len },
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
                .entryPoint = stringView("main"),
                .constantCount = 0,
                .constants = null,
            },
        });
    }
}

fn ensureBuffers(ctx: *GpuContext, cache: *Cache, shape: ValidShape) !void {
    if (cache.m == @as(usize, shape.m) and
        cache.k == @as(usize, shape.k) and
        cache.n == @as(usize, shape.n) and
        cache.bind_group != null)
    {
        return;
    }

    cache.deinitBuffers();
    errdefer cache.deinitBuffers();

    cache.a = try createStorageBuffer(
        ctx,
        shape.a_bytes,
        wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.b = try createStorageBuffer(
        ctx,
        shape.b_bytes,
        wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.c = try createStorageBuffer(
        ctx,
        shape.c_bytes,
        wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc,
    );
    cache.params = try createStorageBuffer(
        ctx,
        @sizeOf(Params),
        wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst,
    );
    cache.staging = try createStorageBuffer(
        ctx,
        shape.c_bytes,
        wgpu.WGPUBufferUsage_MapRead | wgpu.WGPUBufferUsage_CopyDst,
    );

    var entries: [4]wgpu.WGPUBindGroupEntry = undefined;
    for (&entries) |*entry| entry.* = std.mem.zeroes(wgpu.WGPUBindGroupEntry);
    const buffers = [4]wgpu.WGPUBuffer{ cache.a, cache.b, cache.c, cache.params };
    for (0..4) |binding| {
        entries[binding].binding = @intCast(binding);
        entries[binding].buffer = buffers[binding];
        entries[binding].offset = 0;
        entries[binding].size = wgpu.WGPU_WHOLE_SIZE;
    }
    cache.bind_group = try ctx.createBindGroup(&wgpu.WGPUBindGroupDescriptor{
        .nextInChain = null,
        .label = emptyStringView(),
        .layout = cache.bind_group_layout,
        .entryCount = entries.len,
        .entries = entries[0..].ptr,
    });

    cache.m = shape.m;
    cache.k = shape.k;
    cache.n = shape.n;
}

fn runImpl(
    ctx: *GpuContext,
    cache: *Cache,
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
    repetitions: usize,
) !void {
    const shape = validateShape(m, k, n) orelse return error.GpuError;
    if (a.len != m * k or b.len != k * n or out.len != m * n) return error.GpuError;
    if (repetitions == 0) return error.GpuError;
    if (!canRun(ctx.limits, m, k, n, variant)) return error.GpuError;
    const grid = try dispatchGridFor(variant, ctx.limits, shape);

    if (cache.device != ctx.device) {
        cache.deinit();
        cache.device = ctx.device;
    }
    try ensurePipelines(ctx, cache);
    try ensureBuffers(ctx, cache, shape);

    writeBytes(ctx, cache.a, std.mem.sliceAsBytes(a));
    writeBytes(ctx, cache.b, std.mem.sliceAsBytes(b));
    var params = Params{ .m = shape.m, .k = shape.k, .n = shape.n, .pad = 0 };
    writeBytes(ctx, cache.params, std.mem.asBytes(&params));

    const index = variantIndex(variant);
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
        wgpu.wgpuComputePassEncoderSetBindGroup(pass, 0, cache.bind_group, 0, null);
        wgpu.wgpuComputePassEncoderDispatchWorkgroups(pass, grid.x, grid.y, 1);
        wgpu.wgpuComputePassEncoderEnd(pass);
        wgpu.wgpuComputePassEncoderRelease(pass);
    }
    wgpu.wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        cache.c,
        0,
        cache.staging,
        0,
        @intCast(shape.c_bytes),
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

    try ctx.readBuffer(cache.staging, std.mem.sliceAsBytes(out));
}

/// Run one GEMM and read the result back.  Uses the process-local GPU context
/// and the shared resource cache.
pub fn runWithContext(
    ctx: *GpuContext,
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) !void {
    lockCache();
    defer unlockCache();
    return runImpl(ctx, &global_cache, variant, m, k, n, a, b, out, 1);
}

/// Steady-state variant: upload A/B once, run `repetitions` real dispatches,
/// then read C back once.  Used by the benchmark to separate kernel throughput
/// from the per-call upload/readback cost; never used by backend selection.
pub fn runBatchedWithContext(
    ctx: *GpuContext,
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
    repetitions: usize,
) !void {
    lockCache();
    defer unlockCache();
    return runImpl(ctx, &global_cache, variant, m, k, n, a, b, out, repetitions);
}

/// `gemm` through the comptime engine's process-local context, recording the
/// failure reason for the CPU-fallback path.
pub fn gemm(
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) !void {
    context_mod.clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (context_mod.lastFallbackReason() == null) {
            context_mod.recordFallback(@errorName(err));
        }
        return err;
    };
    runWithContext(ctx, variant, m, k, n, a, b, out) catch |err| {
        context_mod.recordFallback(@errorName(err));
        return err;
    };
}

pub fn gemmBatched(
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
    repetitions: usize,
) !void {
    context_mod.clearFallbackReason();
    const ctx = context_mod.global() catch |err| {
        if (context_mod.lastFallbackReason() == null) {
            context_mod.recordFallback(@errorName(err));
        }
        return err;
    };
    runBatchedWithContext(ctx, variant, m, k, n, a, b, out, repetitions) catch |err| {
        context_mod.recordFallback(@errorName(err));
        return err;
    };
}

// ---- CPU reference implementations (used by tests, the CLI and benchmarks) ----

/// C = A * B with the accumulation order k ascending, matching both WGSL
/// kernels.  Row-major throughout.
pub fn referenceScalar(
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) void {
    @memset(out[0 .. m * n], 0);
    for (0..m) |row| {
        for (0..k) |kk| {
            const av = a[row * k + kk];
            const b_row = b[kk * n ..][0..n];
            const out_row = out[row * n ..][0..n];
            for (0..n) |col| {
                out_row[col] += av * b_row[col];
            }
        }
    }
}

pub fn referenceSimd(
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) void {
    const V = @Vector(8, f32);
    @memset(out[0 .. m * n], 0);

    // Register blocking: 4 output rows x 8 columns per inner step.  Keeping the
    // four accumulators in vector registers (instead of reading/writing the
    // output vector for every k) removes the store-to-load dependency and cut
    // the memory traffic per MAC by ~8x versus the naive i-k-j SIMD loop.
    var row: usize = 0;
    while (row < m) : (row += 4) {
        const r0 = row;
        // Clamp the padding rows to the last valid row; their results are
        // discarded below, which keeps the hot loop branch-free.
        const r1 = @min(row + 1, m - 1);
        const r2 = @min(row + 2, m - 1);
        const r3 = @min(row + 3, m - 1);

        var col: usize = 0;
        while (col + 8 <= n) : (col += 8) {
            var acc0: V = @splat(0);
            var acc1: V = @splat(0);
            var acc2: V = @splat(0);
            var acc3: V = @splat(0);
            for (0..k) |kk| {
                const bv: V = @as(*align(1) const V, @ptrCast(b.ptr + kk * n + col)).*;
                const a0: V = @splat(a[r0 * k + kk]);
                const a1: V = @splat(a[r1 * k + kk]);
                const a2: V = @splat(a[r2 * k + kk]);
                const a3: V = @splat(a[r3 * k + kk]);
                acc0 += a0 * bv;
                acc1 += a1 * bv;
                acc2 += a2 * bv;
                acc3 += a3 * bv;
            }
            @as(*align(1) V, @ptrCast(out.ptr + r0 * n + col)).* = acc0;
            if (row + 1 < m) @as(*align(1) V, @ptrCast(out.ptr + r1 * n + col)).* = acc1;
            if (row + 2 < m) @as(*align(1) V, @ptrCast(out.ptr + r2 * n + col)).* = acc2;
            if (row + 3 < m) @as(*align(1) V, @ptrCast(out.ptr + r3 * n + col)).* = acc3;
        }

        // Column tail (n % 8), accumulated in the same k order.
        while (col < n) : (col += 1) {
            var sum0: f32 = 0;
            var sum1: f32 = 0;
            var sum2: f32 = 0;
            var sum3: f32 = 0;
            for (0..k) |kk| {
                const bv = b[kk * n + col];
                sum0 += a[r0 * k + kk] * bv;
                sum1 += a[r1 * k + kk] * bv;
                sum2 += a[r2 * k + kk] * bv;
                sum3 += a[r3 * k + kk] * bv;
            }
            out[r0 * n + col] = sum0;
            if (row + 1 < m) out[r1 * n + col] = sum1;
            if (row + 2 < m) out[r2 * n + col] = sum2;
            if (row + 3 < m) out[r3 * n + col] = sum3;
        }
    }
}

/// Largest absolute element-wise difference; the CLI prints this instead of a
/// bare "OK" so the float-order tolerance is visible.
pub fn maxAbsDiff(expected: []const f32, actual: []const f32) f32 {
    var max_diff: f32 = 0;
    for (expected, actual) |e, value| {
        max_diff = @max(max_diff, @abs(e - value));
    }
    return max_diff;
}

// ---- tests ----

const test_shapes = [_][3]usize{
    .{ 1, 1, 1 },
    .{ 2, 3, 5 },
    .{ 7, 9, 11 },
    .{ 13, 17, 19 },
    .{ 37, 53, 61 },
    .{ 64, 64, 64 },
    .{ 100, 90, 80 },
};

fn fillDeterministic(data: []f32, seed: usize) void {
    for (data, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt((index * 31 + seed * 17) % 23)) * 0.25 - 2.0;
    }
}

fn expectClose(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    if (expected.len != actual.len) return error.TestExpectedApproxEq;
    for (expected, actual) |e, value| {
        const scale = @max(1.0, @abs(e));
        if (@abs(e - value) > tolerance * scale) return error.TestExpectedApproxEq;
    }
}

test "gemm cpu scalar and simd references agree on edge shapes" {
    const gpa = std.testing.allocator;
    for (test_shapes) |shape| {
        const m = shape[0];
        const k = shape[1];
        const n = shape[2];
        const a = try gpa.alloc(f32, m * k);
        defer gpa.free(a);
        const b = try gpa.alloc(f32, k * n);
        defer gpa.free(b);
        const out_scalar = try gpa.alloc(f32, m * n);
        defer gpa.free(out_scalar);
        const out_simd = try gpa.alloc(f32, m * n);
        defer gpa.free(out_simd);

        fillDeterministic(a, 1);
        fillDeterministic(b, 2);
        referenceScalar(m, k, n, a, b, out_scalar);
        referenceSimd(m, k, n, a, b, out_simd);
        try expectClose(out_scalar, out_simd, 1e-5);
    }
}

test "gemm gpu simple and tiled variants match the cpu reference" {
    const gpa = std.testing.allocator;

    var context = GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();
    defer resetCache();

    // Small shapes only: this test is about correctness, not throughput.
    const shapes = [_][3]usize{
        .{ 1, 1, 1 },
        .{ 2, 3, 5 },
        .{ 13, 17, 19 },
        .{ 64, 64, 64 },
        .{ 100, 90, 80 },
    };
    var verified: usize = 0;
    for (shapes) |shape| {
        const m = shape[0];
        const k = shape[1];
        const n = shape[2];
        const a = try gpa.alloc(f32, m * k);
        defer gpa.free(a);
        const b = try gpa.alloc(f32, k * n);
        defer gpa.free(b);
        const reference = try gpa.alloc(f32, m * n);
        defer gpa.free(reference);
        const actual = try gpa.alloc(f32, m * n);
        defer gpa.free(actual);

        fillDeterministic(a, 3);
        fillDeterministic(b, 4);
        referenceScalar(m, k, n, a, b, reference);

        for ([_]Variant{ .simple, .tiled }) |variant| {
            if (!canRun(context.limits, m, k, n, variant)) continue;
            @memset(actual, 0);
            try runWithContext(&context, variant, m, k, n, a, b, actual);
            try expectClose(reference, actual, 1e-4);
            verified += 1;
        }
    }
    try std.testing.expect(verified > 0);
}

test "gemm batched long gpu work does not expire the readback wait" {
    // Regression: the readback wait used to be capped at a fixed number of
    // ProcessEvents calls, which expired while a slow multi-dispatch batch was
    // still running.  The pending mapping then made the next submit fail with
    // wgpu-native's fatal "buffer is still mapped" validation error.  The wait
    // is now a wall-clock budget and a timeout cancels the mapping.
    const gpa = std.testing.allocator;

    var context = GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();
    defer resetCache();

    const m = 1024;
    const k = 1024;
    const n = 1024;
    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, k * n);
    defer gpa.free(b);
    const batched = try gpa.alloc(f32, m * n);
    defer gpa.free(batched);
    const single = try gpa.alloc(f32, m * n);
    defer gpa.free(single);
    fillDeterministic(a, 5);
    fillDeterministic(b, 6);

    // Six slow simple-kernel dispatches keep the GPU busy long enough that a
    // fixed pump-count budget would have expired before the map completed.
    try runBatchedWithContext(&context, .simple, m, k, n, a, b, batched, 6);
    // The next submit reuses the same staging buffer; under the old behavior
    // this is where the fatal "still mapped" validation error appeared.
    try runWithContext(&context, .tiled, m, k, n, a, b, single);
    try expectClose(batched, single, 1e-4);
}

test "gemm canRun rejects shapes outside device limits" {
    const limits = context_mod.GpuLimits{
        .maxComputeWorkgroupsPerDimension = 65_535,
        .maxStorageBufferBindingSize = 4096,
        .maxBufferSize = 8192,
    };

    // 32x32x32 f32 = 4 KiB per operand, fits the storage limit.
    try std.testing.expect(canRun(limits, 32, 32, 32, .simple));
    try std.testing.expect(canRun(limits, 32, 32, 32, .tiled));

    // 64x64x64 needs 16 KiB operands, above maxStorageBufferBindingSize.
    try std.testing.expect(!canRun(limits, 64, 64, 64, .simple));

    // Tiled grid columns: n / 16 must fit maxComputeWorkgroupsPerDimension.
    var tiny_dimension = limits;
    tiny_dimension.maxComputeWorkgroupsPerDimension = 1;
    tiny_dimension.maxStorageBufferBindingSize = 1 << 30;
    tiny_dimension.maxBufferSize = 1 << 30;
    try std.testing.expect(canRun(tiny_dimension, 16, 16, 16, .tiled));
    try std.testing.expect(!canRun(tiny_dimension, 16, 16, 32, .tiled));
    try std.testing.expect(!canRun(tiny_dimension, 32, 16, 16, .tiled));

    // Zero-sized dimensions never run.
    try std.testing.expect(!canRun(limits, 0, 32, 32, .simple));
}
