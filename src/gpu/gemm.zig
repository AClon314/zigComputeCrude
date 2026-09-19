//! GEMM (A[m x k] * B[k x n] = C[m x n]) — Step 3 of the GPU roadmap,
//! migrated to the M0 runtime (M1).
//!
//! Two variants:
//!
//!   * `simple` — one invocation per output element, global memory only.  This
//!     is the correctness baseline (and deliberately the slow one).
//!   * `tiled`  — a 16x16 output tile per 64-invocation workgroup with the A/B
//!     blocks staged in workgroup memory, so each input element is read from
//!     global memory once per tile row/column instead of once per output.
//!
//! `Kernels` is the composable half: compiled pipelines plus a `bind` helper,
//! usable inside any `runtime.Chain` with caller-owned buffers (this is what
//! the heterogeneous chain demo uses).  The slice-based `gemm`/`gemmBatched`
//! entry points keep the historical per-call behaviour (upload, dispatch,
//! readback) on top of a device+shape buffer cache.

const std = @import("std");
const determinism = @import("../determinism.zig");
const runtime = @import("../runtime.zig");
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

const kernel_bindings = [_]runtime.Binding{
    .{ .kind = .storage, .access = .read }, // a
    .{ .kind = .storage, .access = .read }, // b
    .{ .kind = .storage, .access = .write }, // c
    .{ .kind = .uniform, .access = .read }, // params
};

/// Compiled pipelines for one device; buffer-agnostic and cheap to share.
pub const Kernels = struct {
    ctx: *runtime.Device,
    simple: runtime.Kernel,
    tiled: runtime.Kernel,

    pub fn init(ctx: *runtime.Device) !Kernels {
        var self = Kernels{
            .ctx = ctx,
            .simple = try runtime.Kernel.init(ctx, simple_shader, "main", &kernel_bindings, 64),
            .tiled = undefined,
        };
        errdefer self.simple.deinit();
        self.tiled = try runtime.Kernel.init(ctx, tiled_shader, "main", &kernel_bindings, 64);
        return self;
    }

    pub fn deinit(self: *Kernels) void {
        self.tiled.deinit();
        self.simple.deinit();
    }

    pub fn kernel(self: *const Kernels, variant: Variant) *const runtime.Kernel {
        return if (variant == .simple) &self.simple else &self.tiled;
    }

    /// Bind group for `(a, b, c, params)` in that order.
    pub fn bind(
        self: *const Kernels,
        variant: Variant,
        a: *const runtime.Buffer,
        b: *const runtime.Buffer,
        c: *const runtime.Buffer,
        params: *const runtime.Buffer,
    ) !?*anyopaque {
        return self.kernel(variant).createBindGroup(&.{ a, b, c, params });
    }
};

const VariantCache = struct {
    ctx: *runtime.Device,
    shape: ValidShape,
    a: runtime.Buffer,
    b: runtime.Buffer,
    c: runtime.Buffer,
    params: runtime.Buffer,
    binds: [2]?*anyopaque,

    fn init(ctx: *runtime.Device, shape: ValidShape) !VariantCache {
        var self = VariantCache{
            .ctx = ctx,
            .shape = shape,
            .a = undefined,
            .b = undefined,
            .c = undefined,
            .params = undefined,
            .binds = .{ null, null },
        };
        self.a = try runtime.Buffer.init(ctx, shape.a_bytes, runtime.buffer.storage_r);
        errdefer self.a.deinit(ctx);
        self.b = try runtime.Buffer.init(ctx, shape.b_bytes, runtime.buffer.storage_r);
        errdefer self.b.deinit(ctx);
        self.c = try runtime.Buffer.init(ctx, shape.c_bytes, runtime.buffer.storage_rw);
        errdefer self.c.deinit(ctx);
        self.params = try runtime.Buffer.init(ctx, @sizeOf(Params), runtime.buffer.uniform);
        errdefer self.params.deinit(ctx);
        return self;
    }

    fn deinit(self: *VariantCache) void {
        for (&self.binds) |*handle| {
            if (handle.*) |bind| runtime.releaseBindGroup(bind);
            handle.* = null;
        }
        self.params.deinit(self.ctx);
        self.c.deinit(self.ctx);
        self.b.deinit(self.ctx);
        self.a.deinit(self.ctx);
    }
};

var kernels_cache: ?Kernels = null;
var shape_cache: ?VariantCache = null;
var cache_mutex: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!cache_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn unlock() void {
    cache_mutex.unlock();
}

/// Drop cached pipelines/buffers (tests; also after a device change).
pub fn resetCache() void {
    lock();
    defer unlock();
    if (shape_cache) |*cache| cache.deinit();
    shape_cache = null;
    if (kernels_cache) |*kernels| kernels.deinit();
    kernels_cache = null;
}

fn variantIndex(variant: Variant) usize {
    return switch (variant) {
        .simple => 0,
        .tiled => 1,
    };
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
    return gridIsValid(limits, shape, variant);
}

fn gridIsValid(limits: context_mod.GpuLimits, shape: ValidShape, variant: Variant) bool {
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

/// Dispatch grid for a variant/shape (2D-flattened for `simple`, output tiles
/// for `tiled`).  Exposed so callers composing their own chains can reuse it.
pub fn gridFor(
    limits: context_mod.GpuLimits,
    m: usize,
    k: usize,
    n: usize,
    variant: Variant,
) !context_mod.WorkgroupGrid {
    const shape = validateShape(m, k, n) orelse return error.GpuError;
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

fn ensureKernels(ctx: *runtime.Device) !void {
    if (kernels_cache) |*kernels| {
        if (kernels.ctx.device == ctx.device) return;
        kernels.deinit();
        kernels_cache = null;
        // Bind groups in the shape cache reference the old layouts.
        if (shape_cache) |*cache| cache.deinit();
        shape_cache = null;
    }
    kernels_cache = try Kernels.init(ctx);
}

fn ensureShape(ctx: *runtime.Device, shape: ValidShape) !*VariantCache {
    if (shape_cache) |*cache| {
        if (cache.ctx.device == ctx.device and
            cache.shape.m == shape.m and cache.shape.k == shape.k and cache.shape.n == shape.n)
        {
            return cache;
        }
        cache.deinit();
        shape_cache = null;
    }

    var cache = try VariantCache.init(ctx, shape);
    errdefer cache.deinit();
    const kernels = &kernels_cache.?;
    cache.binds[0] = try kernels.bind(.simple, &cache.a, &cache.b, &cache.c, &cache.params);
    cache.binds[1] = try kernels.bind(.tiled, &cache.a, &cache.b, &cache.c, &cache.params);
    shape_cache = cache;
    return &shape_cache.?;
}

fn runImpl(
    ctx: *runtime.Device,
    variant: Variant,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
    iterations: usize,
) !void {
    const shape = validateShape(m, k, n) orelse return error.GpuError;
    if (a.len != m * k or b.len != k * n or out.len != m * n) return error.GpuError;
    if (iterations == 0) return error.GpuError;
    if (!canRun(ctx.limits, m, k, n, variant)) return error.GpuError;

    lock();
    defer unlock();

    try ensureKernels(ctx);
    const cache = try ensureShape(ctx, shape);

    cache.a.markHostDirty();
    try cache.a.toDevice(ctx, std.mem.sliceAsBytes(a));
    cache.b.markHostDirty();
    try cache.b.toDevice(ctx, std.mem.sliceAsBytes(b));
    var params = Params{ .m = shape.m, .k = shape.k, .n = shape.n, .pad = 0 };
    cache.params.markHostDirty();
    try cache.params.toDevice(ctx, std.mem.asBytes(&params));

    const grid = try gridFor(ctx.limits, m, k, n, variant);
    const kernel = kernels_cache.?.kernel(variant);
    const buffers = [4]*runtime.Buffer{ &cache.a, &cache.b, &cache.c, &cache.params };

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    for (0..iterations) |_| {
        try chain.dispatch(kernel, cache.binds[variantIndex(variant)].?, &buffers, grid);
    }
    try chain.download(&cache.c, std.mem.sliceAsBytes(out));
    try chain.submit();
}

/// Run one GEMM and read the result back.  Uses the process-local GPU context.
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
    return runImpl(ctx, variant, m, k, n, a, b, out, 1);
}

/// Steady-state variant: upload A/B once, run `repetitions` real dispatches,
/// then read C back once.  Used by the benchmark to separate kernel throughput
/// from the per-call upload/readback cost.
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
    return runImpl(ctx, variant, m, k, n, a, b, out, repetitions);
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

/// CPU 侧寄存器分块的行数 —— **comptime 旋钮**（不是运行时参数）。
///
/// 内层是 i-k-j：每读一行 B 的 `width` 个元素，就服务 ROWS 个输出行；B 的读取量因此是
/// `(m/ROWS) * k * n`，ROWS 直接决定"每字节 B 换来多少次 FMA"。实测（Ryzen 5 5600H /
/// AVX2，W=8，64-bit 三跑取最大，prototype 见 commit message）：
///
///   ROWS    256³     512³     1024³
///    4     54.8     47.1     25.5   GFLOP/s
///    6     74.2     62.8     42.3
///    8     86.2     72.8     51.4   ← 选它
///   10     92.0     70.7     52.4
///   12     89.4     80.7     39.5   （寄存器溢出，1024³ 反弹）
///
/// 8 在三个尺寸上都稳（1.6~2.0x vs ROWS=4），10/12 收益不稳定；ROWS × width 个向量
/// 累加器要放进 16 个 AVX2 寄存器，所以不能无限加。`noalias` 实测无差别（±1%），
/// 故不加——那会引入"输出不得与输入别名"的隐性契约。
pub const cpu_block_rows: usize = 8;

/// C = A * B，行主序，累加顺序 k 升序（与两个 WGSL kernel 一致）。
/// 每行 ROWS 个输出行并行累加；**舍入顺序与 ROWS 无关**（每个输出元素仍是 k 升序
/// `@mulAdd`），所以这个旋钮不影响对拍结果，只影响速度。
pub fn referenceSimd(
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) void {
    referenceSimdBlocked(cpu_block_rows, m, k, n, a, b, out);
}

/// `referenceSimd` 的 comptime 分块版本。ROWS 是编译期常量 → 累加器全部静态展开
/// 到寄存器，内层没有数组索引/边界分支（这正是 comptime 在这里买到的东西）。
pub fn referenceSimdBlocked(
    comptime ROWS: usize,
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    out: []f32,
) void {
    comptime if (ROWS == 0 or ROWS > 16) @compileError("ROWS out of range");
    const width = vectorWidth();
    const V = @Vector(width, f32);
    @memset(out[0 .. m * n], 0);

    // Register blocking: ROWS output rows x `width` columns per inner step.  Keeping
    // the accumulators in vector registers (instead of reading/writing the output
    // vector for every k) removes the store-to-load dependency and cuts the memory
    // traffic per MAC by roughly ROWS x versus the naive i-k-j SIMD loop.
    var row: usize = 0;
    while (row < m) : (row += ROWS) {
        // Clamp the padding rows to the last valid row; their results are
        // discarded below, which keeps the hot loop branch-free.
        var rows_index: [ROWS]usize = undefined;
        var rows_valid: [ROWS]bool = undefined;
        inline for (0..ROWS) |r| {
            rows_index[r] = @min(row + r, m - 1);
            rows_valid[r] = row + r < m;
        }

        var col: usize = 0;
        while (col + width <= n) : (col += width) {
            var acc: [ROWS]V = undefined;
            inline for (0..ROWS) |r| acc[r] = @splat(0);
            for (0..k) |kk| {
                const bv: V = @as(*align(1) const V, @ptrCast(b.ptr + kk * n + col)).*;
                inline for (0..ROWS) |r| {
                    const av: V = @splat(a[rows_index[r] * k + kk]);
                    // Explicit @mulAdd: Zig strict FP does not contract a*b+c into
                    // FMA, and without it the inner loop is 2x the arithmetic
                    // instructions (vmulps+vaddps instead of vfmadd).
                    acc[r] = @mulAdd(V, av, bv, acc[r]);
                }
            }
            inline for (0..ROWS) |r| {
                if (rows_valid[r]) {
                    @as(*align(1) V, @ptrCast(out.ptr + rows_index[r] * n + col)).* = acc[r];
                }
            }
        }

        // Column tail (n % width), accumulated in the same k order.
        while (col < n) : (col += 1) {
            var sums: [ROWS]f32 = undefined;
            inline for (0..ROWS) |r| sums[r] = 0;
            for (0..k) |kk| {
                const bv = b[kk * n + col];
                inline for (0..ROWS) |r| {
                    sums[r] = @mulAdd(f32, a[rows_index[r] * k + kk], bv, sums[r]);
                }
            }
            inline for (0..ROWS) |r| {
                if (rows_valid[r]) out[rows_index[r] * n + col] = sums[r];
            }
        }
    }
}

/// Target-dependent vector width (AVX2 = 8 f32, AVX-512 = 16, NEON = 4, …).
fn vectorWidth() comptime_int {
    return std.simd.suggestVectorLength(f32) orelse 4;
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

/// GPU 路径的统一判据（tolerant/accumulated = 1e-4，见 determinism.zig 的表）。
fn expectClose(expected: []const f32, actual: []const f32) !void {
    return determinism.expectSlices(.tolerant, .accumulated, f32, expected, actual);
}

/// CPU scalar 与 CPU SIMD 只差累加顺序，既有判据是 1e-5，比 GPU 路径更紧。
fn expectCloseSameBackendFamily(expected: []const f32, actual: []const f32) !void {
    return determinism.expectSlicesWithin(
        .{ .absolute = 1e-5, .relative = 1e-5 },
        f32,
        expected,
        actual,
    );
}

test "referenceSimdBlocked is bit-identical across register block sizes" {
    // ROWS 只改变"同时算几个输出行"，不改变每个输出元素的 k 升序 @mulAdd 顺序，
    // 所以不同 ROWS 必须逐位一致（这也是它敢做默认旋钮的前提）。
    const gpa = std.testing.allocator;
    for (test_shapes) |shape| {
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
        referenceSimdBlocked(4, m, k, n, a, b, reference);

        inline for ([_]usize{ 1, 2, 3, 8, 16 }) |rows| {
            @memset(actual, 0);
            referenceSimdBlocked(rows, m, k, n, a, b, actual);
            try std.testing.expectEqualSlices(f32, reference, actual);
        }
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
        try expectCloseSameBackendFamily(out_scalar, out_simd);
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
            try expectClose(reference, actual);
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
    try expectClose(batched, single);
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
