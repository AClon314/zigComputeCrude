const std = @import("std");
const backend = @import("backend.zig");
const gpu_pipeline = @import("gpu/pipeline.zig");
const BackendType = backend.BackendType;

/// 基于 comptime 的静态派发引擎。选择 .cpu_simd 时生成的机器码只含 SIMD 循环。
pub fn ComputeEngine(comptime bt: BackendType) type {
    return struct {
        pub const backend: BackendType = bt;

        pub fn name() []const u8 {
            return bt.name();
        }

        /// out[i] = a[i] + b[i]
        pub fn add(comptime T: type, out: []T, a: []const T, b: []const T) void {
            std.debug.assert(a.len == b.len and b.len == out.len);
            switch (bt) {
                .cpu_scalar => addScalar(T, out, a, b),
                .cpu_simd => addSimd(T, out, a, b),
                .gpu_webgpu => if (T == f32) {
                    gpu_pipeline.add(out, a, b) catch addSimd(T, out, a, b);
                } else {
                    gpu_pipeline.recordFallback("gpu_webgpu is only available for f32; used cpu_simd");
                    addSimd(T, out, a, b);
                },
                .gpu_cuda => @panic("computeAccel: gpu_cuda backend not implemented"),
            }
        }

        /// out[i] = alpha * x[i] + y[i]
        pub fn saxpy(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
            std.debug.assert(x.len == y.len and y.len == out.len);
            switch (bt) {
                .cpu_scalar => saxpyScalar(T, alpha, out, x, y),
                .cpu_simd => saxpySimd(T, alpha, out, x, y),
                .gpu_webgpu => if (T == f32) {
                    gpu_pipeline.saxpy(alpha, out, x, y) catch saxpySimd(T, alpha, out, x, y);
                } else {
                    gpu_pipeline.recordFallback("gpu_webgpu is only available for f32; used cpu_simd");
                    saxpySimd(T, alpha, out, x, y);
                },
                .gpu_cuda => @panic("computeAccel: gpu_cuda backend not implemented"),
            }
        }
    };
}

// ---- 内核实现（可放在文件底部，为 file-private fn）----

/// 目标相关的最优向量宽度（AVX2=8、AVX-512=16、NEON=4、wasm simd128=4…）。
/// 不用写死 8：写死会在非 AVX2 平台上浪费（或拆寄存器）。
fn vectorWidth(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 4;
}

fn addScalar(comptime T: type, out: []T, a: []const T, b: []const T) void {
    for (0..a.len) |i| out[i] = a[i] + b[i];
}

fn addSimd(comptime T: type, out: []T, a: []const T, b: []const T) void {
    const width = vectorWidth(T);
    const V = @Vector(width, T);
    var i: usize = 0;
    const chunks = a.len / width;
    while (i < chunks * width) : (i += width) {
        const va: V = @as(*align(1) const V, @ptrCast(a.ptr + i)).*;
        const vb: V = @as(*align(1) const V, @ptrCast(b.ptr + i)).*;
        const vr: V = va + vb;
        @as(*align(1) V, @ptrCast(out.ptr + i)).* = vr;
    }
    while (i < a.len) : (i += 1) out[i] = a[i] + b[i];
}

fn saxpyScalar(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
    for (0..x.len) |i| out[i] = alpha * x[i] + y[i];
}

fn saxpySimd(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
    const width = vectorWidth(T);
    const V = @Vector(width, T);
    const va: V = @splat(alpha);
    var i: usize = 0;
    const chunks = x.len / width;
    while (i < chunks * width) : (i += width) {
        const vx: V = @as(*align(1) const V, @ptrCast(x.ptr + i)).*;
        const vy: V = @as(*align(1) const V, @ptrCast(y.ptr + i)).*;
        // @mulAdd 而不是 vx * va + vy：Zig 默认严格浮点不会把 mul+add 收缩成 FMA，
        // 显式 @mulAdd 才能让 x86 生成 vfmadd（对 saxpy/GEMM 是实打实的减半算术指令）。
        const vr: V = @mulAdd(V, va, vx, vy);
        @as(*align(1) V, @ptrCast(out.ptr + i)).* = vr;
    }
    while (i < x.len) : (i += 1) out[i] = alpha * x[i] + y[i];
}

test "engine add correctness across backends" {
    const n = 1009; // 非 8 的倍数，验证尾部处理
    const a = [_]f32{2.0} ** n;
    const b = [_]f32{3.0} ** n;
    var outs: [n]f32 = undefined;
    var outv: [n]f32 = undefined;

    ComputeEngine(.cpu_scalar).add(f32, &outs, &a, &b);
    ComputeEngine(.cpu_simd).add(f32, &outv, &a, &b);
    try std.testing.expectEqualSlices(f32, &outs, &outv);
    try std.testing.expectEqual(@as(f32, 5.0), outs[0]);
    try std.testing.expectEqual(@as(f32, 5.0), outs[n - 1]);
}

test "engine saxpy correctness across backends" {
    const n = 1027;
    const x = [_]f32{2.0} ** n;
    const y = [_]f32{1.0} ** n;
    var outs: [n]f32 = undefined;
    var outv: [n]f32 = undefined;

    ComputeEngine(.cpu_scalar).saxpy(f32, 3.0, &outs, &x, &y); // 3*2+1 = 7
    ComputeEngine(.cpu_simd).saxpy(f32, 3.0, &outv, &x, &y);
    try std.testing.expectEqualSlices(f32, &outs, &outv);
    try std.testing.expectEqual(@as(f32, 7.0), outs[0]);
    try std.testing.expectEqual(@as(f32, 7.0), outs[n - 1]);
}
