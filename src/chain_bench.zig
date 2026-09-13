//! M0 ablation benchmarks for the runtime (`Buffer`/`Kernel`/`Chain`).
//!
//! Design rule: the three modes compute *exactly the same thing*; the only
//! difference is how many host round trips happen.  That isolates what
//! residency and chaining actually buy:
//!
//!   * `per_call`   — upload + dispatch + readback per step (the old API shape);
//!   * `per_submit` — resident buffers, one submit per step, one readback total;
//!   * `chained`    — resident buffers, one submit and one readback for the
//!                    whole chain (optionally several different kernels).
//!
//! Both workloads are compared against a CPU SIMD reference, so "faster" is
//! always "faster with identical results".

const std = @import("std");
const runtime = @import("runtime.zig");
const wgpu = @import("gpu/webgpu.zig");
const bench = @import("bench.zig");
const engine = @import("engine.zig");
const context_mod = @import("gpu/context.zig");
const gemm_mod = @import("gpu/gemm.zig");
const reduce_mod = @import("gpu/reduce.zig");

const saxpy_shader = @embedFile("gpu/shaders/saxpy.wgsl");
const bias_add_shader = @embedFile("gpu/shaders/bias_add.wgsl");

pub const Mode = enum {
    per_call,
    per_submit,
    chained,

    pub fn name(self: Mode) []const u8 {
        return switch (self) {
            .per_call => "per_call",
            .per_submit => "per_submit",
            .chained => "chained",
        };
    }
};

pub const SaxpyResult = struct {
    total_ns: u64,
    gbps: f64,
    max_diff: f32,
};

const AlphaParams = extern struct {
    alpha: f32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const storage_rw = runtime.buffer.storage_rw;
const storage_r = runtime.buffer.storage_r;

/// CPU reference for `x = alpha * x + y` applied `steps` times.
pub fn referenceSaxpyChain(x: []f32, y: []const f32, alpha: f32, steps: usize) void {
    for (0..steps) |_| {
        for (0..x.len) |i| x[i] = alpha * x[i] + y[i];
    }
}

/// CPU SIMD timing of the same chain (in-place, so no extra traffic).
pub fn timeCpuSimdSaxpyChain(x: []f32, y: []const f32, alpha: f32, steps: usize) u64 {
    const t0 = bench.nowNs();
    for (0..steps) |_| engine.ComputeEngine(.cpu_simd).saxpy(f32, alpha, x, x, y);
    return @intCast(@max(0, bench.nowNs() - t0));
}

/// Run the saxpy chain on the GPU in the requested mode and verify the result
/// against the CPU reference.
pub fn runSaxpyChain(
    allocator: std.mem.Allocator,
    ctx: *runtime.Device,
    n: usize,
    steps: usize,
    mode: Mode,
    alpha: f32,
    initial: []const f32,
    y: []const f32,
    out: []f32,
) !SaxpyResult {
    if (n == 0 or steps == 0) return error.GpuError;
    if (initial.len != n or y.len != n or out.len != n) return error.GpuError;

    const byte_size = n * @sizeOf(f32);
    var kernel = try runtime.Kernel.init(ctx, saxpy_shader, "main", &.{
        .{ .kind = .storage, .access = .read },
        .{ .kind = .storage, .access = .read },
        .{ .kind = .storage, .access = .write },
        .{ .kind = .uniform, .access = .read },
    }, 64);
    defer kernel.deinit();

    var x0 = try runtime.Buffer.init(ctx, byte_size, storage_rw);
    defer x0.deinit(ctx);
    var x1 = try runtime.Buffer.init(ctx, byte_size, storage_rw);
    defer x1.deinit(ctx);
    var y_buf = try runtime.Buffer.init(ctx, byte_size, storage_r);
    defer y_buf.deinit(ctx);
    var params = try runtime.Buffer.init(ctx, @sizeOf(AlphaParams), wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst);
    defer params.deinit(ctx);

    const alpha_value = AlphaParams{ .alpha = alpha };
    try y_buf.toDevice(ctx, std.mem.sliceAsBytes(y));
    params.markHostDirty();
    try params.toDevice(ctx, std.mem.asBytes(&alpha_value));

    const bind_in0 = try kernel.createBindGroup(&.{ &x0, &y_buf, &x1, &params });
    defer runtime.releaseBindGroup(bind_in0);
    const bind_in1 = try kernel.createBindGroup(&.{ &x1, &y_buf, &x0, &params });
    defer runtime.releaseBindGroup(bind_in1);

    const grid = try kernel.gridLinear(n);
    const work = try allocator.alloc(f32, n);
    defer allocator.free(work);
    @memcpy(work, initial);

    const t0 = bench.nowNs();
    switch (mode) {
        .per_call => {
            // Old API shape: every step is a full round trip through the host.
            for (0..steps) |_| {
                x0.markHostDirty();
                try x0.toDevice(ctx, std.mem.sliceAsBytes(work));
                var chain = try runtime.Chain.begin(ctx);
                errdefer chain.deinit();
                try chain.dispatch(&kernel, bind_in0, &.{ &x0, &y_buf, &x1, &params }, grid);
                try chain.download(&x1, std.mem.sliceAsBytes(out));
                try chain.submit();
                @memcpy(work, out);
            }
        },
        .per_submit => {
            // Resident, but the device is drained after every step.
            x0.markHostDirty();
            try x0.toDevice(ctx, std.mem.sliceAsBytes(work));
            for (0..steps) |step| {
                var chain = try runtime.Chain.begin(ctx);
                errdefer chain.deinit();
                if (step % 2 == 0) {
                    try chain.dispatch(&kernel, bind_in0, &.{ &x0, &y_buf, &x1, &params }, grid);
                } else {
                    try chain.dispatch(&kernel, bind_in1, &.{ &x1, &y_buf, &x0, &params }, grid);
                }
                try chain.submit();
            }
            const final = if (steps % 2 == 0) &x0 else &x1;
            try final.toHost(ctx, std.mem.sliceAsBytes(out));
        },
        .chained => {
            // Resident + one submit + one readback for the whole chain.
            x0.markHostDirty();
            try x0.toDevice(ctx, std.mem.sliceAsBytes(work));
            var chain = try runtime.Chain.begin(ctx);
            errdefer chain.deinit();
            for (0..steps) |step| {
                if (step % 2 == 0) {
                    try chain.dispatch(&kernel, bind_in0, &.{ &x0, &y_buf, &x1, &params }, grid);
                } else {
                    try chain.dispatch(&kernel, bind_in1, &.{ &x1, &y_buf, &x0, &params }, grid);
                }
            }
            const final = if (steps % 2 == 0) &x0 else &x1;
            try chain.download(final, std.mem.sliceAsBytes(out));
            try chain.submit();
        },
    }
    const total_ns: u64 = @intCast(@max(0, bench.nowNs() - t0));

    // Verify against the CPU reference on a separate copy (max|diff|).
    const reference = try allocator.alloc(f32, n);
    defer allocator.free(reference);
    @memcpy(reference, initial);
    referenceSaxpyChain(reference, y, alpha, steps);
    const max_diff = maxAbsDiff(reference, out);

    const bytes = @as(f64, @floatFromInt(n * @sizeOf(f32) * 3 * steps));
    return .{
        .total_ns = total_ns,
        .gbps = bytes / @as(f64, @floatFromInt(total_ns)),
        .max_diff = max_diff,
    };
}

fn maxAbsDiff(expected: []const f32, actual: []const f32) f32 {
    var diff: f32 = 0;
    for (expected, actual) |e, value| diff = @max(diff, @abs(e - value));
    return diff;
}

// ---- heterogeneous chain: C = A*B (tiled) -> D = C + bias -> sum(D) ----

pub const PipelineMode = enum {
    /// Each stage is its own submit + host readback (old per-kernel API shape).
    staged,
    /// One submit and one readback for the whole 4-dispatch pipeline.
    chained,
};

pub const PipelineResult = struct {
    total_ns: u64,
    /// CPU reference value (for the relative tolerance).
    reference: f32,
    /// |reference - gpu| in absolute terms.
    max_diff: f32,

    pub fn relativeDiff(self: PipelineResult) f64 {
        const scale: f64 = @max(1.0, @abs(@as(f64, self.reference)));
        return @as(f64, self.max_diff) / scale;
    }
};

/// CPU reference for the whole pipeline (returns the final sum).
pub fn referencePipeline(
    m: usize,
    k: usize,
    n: usize,
    a: []const f32,
    b: []const f32,
    bias: []const f32,
    scratch_c: []f32,
) f32 {
    gemm_mod.referenceSimd(m, k, n, a, b, scratch_c);
    for (0..m) |row| {
        for (0..n) |col| scratch_c[row * n + col] += bias[col];
    }
    return reduce_mod.referenceSumSimd(scratch_c);
}

pub fn runGemmBiasReduceChain(
    allocator: std.mem.Allocator,
    ctx: *runtime.Device,
    m: usize,
    k: usize,
    n: usize,
    mode: PipelineMode,
    repetitions: usize,
    a: []const f32,
    b: []const f32,
    bias: []const f32,
    out_sum: *f32,
) !PipelineResult {
    if (m == 0 or k == 0 or n == 0 or repetitions == 0) return error.GpuError;
    if (a.len != m * k or b.len != k * n or bias.len != n) return error.GpuError;

    const c_elements = m * n;
    const c_bytes = c_elements * @sizeOf(f32);
    const buckets = reduce_mod.bucketCount(ctx.limits, c_elements);
    if (buckets == 0) return error.GpuError;

    var gemm_kernels = try gemm_mod.Kernels.init(ctx);
    defer gemm_kernels.deinit();
    var bias_kernel = try runtime.Kernel.init(ctx, bias_add_shader, "main", &.{
        .{ .kind = .storage, .access = .read },
        .{ .kind = .storage, .access = .read },
        .{ .kind = .storage, .access = .write },
        .{ .kind = .uniform, .access = .read },
    }, 64);
    defer bias_kernel.deinit();
    var reduce_kernels = try reduce_mod.Kernels.init(ctx);
    defer reduce_kernels.deinit();

    var a_buf = try runtime.Buffer.init(ctx, m * k * @sizeOf(f32), storage_r);
    defer a_buf.deinit(ctx);
    var b_buf = try runtime.Buffer.init(ctx, k * n * @sizeOf(f32), storage_r);
    defer b_buf.deinit(ctx);
    var c_buf = try runtime.Buffer.init(ctx, c_bytes, storage_rw);
    defer c_buf.deinit(ctx);
    var d_buf = try runtime.Buffer.init(ctx, c_bytes, storage_rw);
    defer d_buf.deinit(ctx);
    var bias_buf = try runtime.Buffer.init(ctx, n * @sizeOf(f32), storage_r);
    defer bias_buf.deinit(ctx);
    var partials = try runtime.Buffer.init(ctx, buckets * @sizeOf(f32), wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc);
    defer partials.deinit(ctx);
    var final = try runtime.Buffer.init(ctx, @sizeOf(f32), wgpu.WGPUBufferUsage_Storage | wgpu.WGPUBufferUsage_CopySrc);
    defer final.deinit(ctx);

    const GemmParams = extern struct { m: u32, k: u32, n: u32, pad: u32 };
    var gemm_params_buf = try runtime.Buffer.init(ctx, 16, wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst);
    defer gemm_params_buf.deinit(ctx);
    const gp = GemmParams{ .m = @intCast(m), .k = @intCast(k), .n = @intCast(n), .pad = 0 };
    try gemm_params_buf.toDevice(ctx, std.mem.asBytes(&gp));

    var bias_params_buf = try runtime.Buffer.init(ctx, 16, wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst);
    defer bias_params_buf.deinit(ctx);
    const BiasParams = extern struct { n: u32, alpha: f32, pad0: u32, pad1: u32 };
    const bp = BiasParams{ .n = @intCast(n), .alpha = 1.0, .pad0 = 0, .pad1 = 0 };
    try bias_params_buf.toDevice(ctx, std.mem.asBytes(&bp));

    const ReduceParams = extern struct { n: u32, pad0: u32, pad1: u32, pad2: u32 };
    var reduce_params_c = try runtime.Buffer.init(ctx, 16, wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst);
    defer reduce_params_c.deinit(ctx);
    const rc = ReduceParams{ .n = @intCast(c_elements), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    try reduce_params_c.toDevice(ctx, std.mem.asBytes(&rc));
    var reduce_params_p = try runtime.Buffer.init(ctx, 16, wgpu.WGPUBufferUsage_Uniform | wgpu.WGPUBufferUsage_CopyDst);
    defer reduce_params_p.deinit(ctx);
    const rp = ReduceParams{ .n = @intCast(buckets), .pad0 = 0, .pad1 = 0, .pad2 = 0 };
    try reduce_params_p.toDevice(ctx, std.mem.asBytes(&rp));

    try a_buf.toDevice(ctx, std.mem.sliceAsBytes(a));
    try b_buf.toDevice(ctx, std.mem.sliceAsBytes(b));
    try bias_buf.toDevice(ctx, std.mem.sliceAsBytes(bias));

    const gemm_bind = try gemm_kernels.bind(.tiled, &a_buf, &b_buf, &c_buf, &gemm_params_buf);
    defer runtime.releaseBindGroup(gemm_bind);
    const bias_bind = try bias_kernel.createBindGroup(&.{ &c_buf, &bias_buf, &d_buf, &bias_params_buf });
    defer runtime.releaseBindGroup(bias_bind);
    var reduce_binding = try reduce_mod.bindAll(
        &reduce_kernels,
        &d_buf,
        &partials,
        &final,
        &reduce_params_c,
        &reduce_params_p,
    );
    defer reduce_mod.releaseBinding(&reduce_binding);

    const gemm_grid = try gemm_mod.gridFor(ctx.limits, m, k, n, .tiled);
    const bias_grid = try bias_kernel.gridLinear(c_elements);
    const reduce_grid1 = context_mod.WorkgroupGrid{ .x = @intCast(buckets), .y = 1 };
    const reduce_grid2 = context_mod.WorkgroupGrid{ .x = 1, .y = 1 };

    const t0 = bench.nowNs();
    switch (mode) {
        .staged => {
            // Every stage drains to the host and is re-uploaded: the exact
            // behaviour of the pre-runtime per-kernel API.
            const host_c = try allocator.alloc(f32, c_elements);
            defer allocator.free(host_c);
            const host_d = try allocator.alloc(f32, c_elements);
            defer allocator.free(host_d);
            for (0..repetitions) |_| {
                {
                    var chain = try runtime.Chain.begin(ctx);
                    errdefer chain.deinit();
                    try chain.dispatch(gemm_kernels.kernel(.tiled), gemm_bind, &.{ &a_buf, &b_buf, &c_buf, &gemm_params_buf }, gemm_grid);
                    try chain.download(&c_buf, std.mem.sliceAsBytes(host_c));
                    try chain.submit();
                }
                c_buf.markHostDirty();
                try c_buf.toDevice(ctx, std.mem.sliceAsBytes(host_c));
                {
                    var chain = try runtime.Chain.begin(ctx);
                    errdefer chain.deinit();
                    try chain.dispatch(&bias_kernel, bias_bind, &.{ &c_buf, &bias_buf, &d_buf, &bias_params_buf }, bias_grid);
                    try chain.download(&d_buf, std.mem.sliceAsBytes(host_d));
                    try chain.submit();
                }
                d_buf.markHostDirty();
                try d_buf.toDevice(ctx, std.mem.sliceAsBytes(host_d));
                var chain = try runtime.Chain.begin(ctx);
                errdefer chain.deinit();
                try chain.dispatch(reduce_kernels.kernel(.sum), reduce_binding.first[0].?, &.{ &d_buf, &partials, &reduce_params_c }, reduce_grid1);
                try chain.dispatch(reduce_kernels.kernel(.sum), reduce_binding.second[0].?, &.{ &partials, &final, &reduce_params_p }, reduce_grid2);
                try chain.download(&final, std.mem.asBytes(out_sum));
                try chain.submit();
            }
        },
        .chained => {
            for (0..repetitions) |_| {
                var chain = try runtime.Chain.begin(ctx);
                errdefer chain.deinit();
                try chain.dispatch(gemm_kernels.kernel(.tiled), gemm_bind, &.{ &a_buf, &b_buf, &c_buf, &gemm_params_buf }, gemm_grid);
                try chain.dispatch(&bias_kernel, bias_bind, &.{ &c_buf, &bias_buf, &d_buf, &bias_params_buf }, bias_grid);
                try chain.dispatch(reduce_kernels.kernel(.sum), reduce_binding.first[0].?, &.{ &d_buf, &partials, &reduce_params_c }, reduce_grid1);
                try chain.dispatch(reduce_kernels.kernel(.sum), reduce_binding.second[0].?, &.{ &partials, &final, &reduce_params_p }, reduce_grid2);
                try chain.download(&final, std.mem.asBytes(out_sum));
                try chain.submit();
            }
        },
    }
    const total_ns: u64 = @intCast(@max(0, bench.nowNs() - t0));

    const scratch = try allocator.alloc(f32, c_elements);
    defer allocator.free(scratch);
    const reference = referencePipeline(m, k, n, a, b, bias, scratch);
    return .{ .total_ns = total_ns, .reference = reference, .max_diff = @abs(reference - out_sum.*) };
}

// ---- tests ----

test "buffer sync state machine" {
    var buffer = runtime.Buffer{};
    try std.testing.expectEqual(runtime.SyncState.host, buffer.sync);
    buffer.markDeviceWritten();
    try std.testing.expectEqual(runtime.SyncState.device, buffer.sync);
    buffer.markHostDirty();
    try std.testing.expectEqual(runtime.SyncState.host, buffer.sync);
}

test "runtime saxpy chain matches the cpu reference in every mode" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const n: usize = 4096;
    const steps: usize = 8;
    const alpha: f32 = 0.75;
    const initial = try gpa.alloc(f32, n);
    defer gpa.free(initial);
    const y = try gpa.alloc(f32, n);
    defer gpa.free(y);
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    for (initial, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index % 17)) * 0.25 - 1.0;
    }
    for (y, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index % 13)) * 0.5 - 2.0;
    }

    for ([_]Mode{ .per_call, .per_submit, .chained }) |mode| {
        const result = try runSaxpyChain(gpa, ctx, n, steps, mode, alpha, initial, y, out);
        try std.testing.expect(result.max_diff <= 1e-4);
        try std.testing.expect(result.total_ns > 0);
    }
}

test "runtime gemm+bias+reduce chain matches the cpu reference" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    // Deliberately not multiples of the 16x16 tile: exercises the tiled GEMM
    // bounds handling inside a chain.
    const m: usize = 37;
    const k: usize = 53;
    const n: usize = 61;
    const a = try gpa.alloc(f32, m * k);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, k * n);
    defer gpa.free(b);
    const bias = try gpa.alloc(f32, n);
    defer gpa.free(bias);
    for (a, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index % 19)) * 0.125 - 1.0;
    }
    for (b, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index % 11)) * 0.25 - 1.25;
    }
    for (bias, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(index % 7)) * 0.5 - 1.5;
    }

    for ([_]PipelineMode{ .staged, .chained }) |mode| {
        var sum: f32 = 0;
        const result = try runGemmBiasReduceChain(gpa, ctx, m, k, n, mode, 3, a, b, bias, &sum);
        try std.testing.expect(result.relativeDiff() <= 1e-4);
        try std.testing.expect(result.total_ns > 0);
    }
}
