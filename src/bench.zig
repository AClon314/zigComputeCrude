const std = @import("std");
const backend = @import("backend.zig");
const buffer_mod = @import("buffer.zig");
const engine = @import("engine.zig");
const gpu_pipeline = @import("gpu/pipeline.zig");
const BackendType = backend.BackendType;
const DeviceBuffer = buffer_mod.DeviceBuffer;
const ComputeEngine = engine.ComputeEngine;

/// 单调时钟（nanosecond），Zig 0.16 用 posix.clock_gettime。
pub fn nowNs() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// 对固定后端跑 iters 次 add，返回总耗时（ns）。
pub fn timeAdd(comptime T: type, comptime bt: BackendType, out: []T, a: []const T, b: []const T, iters: usize) u64 {
    const t0 = nowNs();
    for (0..iters) |_| ComputeEngine(bt).add(T, out, a, b);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// Time native GPU execution without silently converting a GPU error into a
/// CPU result.  The CLI uses this to label a real GPU measurement correctly.
pub fn timeGpuAdd(out: []f32, a: []const f32, b: []const f32, iters: usize) !u64 {
    const t0 = nowNs();
    for (0..iters) |_| try gpu_pipeline.add(out, a, b);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// Steady-state comparison: the same inputs are uploaded once, then `iters`
/// dispatches are submitted before one staging readback.
pub fn timeGpuAddBatched(out: []f32, a: []const f32, b: []const f32, iters: usize) !u64 {
    const t0 = nowNs();
    try gpu_pipeline.addBatched(out, a, b, iters);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// 运行时 bench 每个可用后端，返回最快的。
pub fn pickBest(allocator: std.mem.Allocator, comptime T: type, size: usize, iters: usize) !BackendType {
    const a = try allocator.alloc(T, size);
    defer allocator.free(a);
    const b = try allocator.alloc(T, size);
    defer allocator.free(b);
    const out = try allocator.alloc(T, size);
    defer allocator.free(out);
    @memset(a, 2.0);
    @memset(b, 3.0);

    var best: ?BackendType = null;
    var best_ns: u64 = std.math.maxInt(u64);
    inline for (.{
        BackendType.cpu_scalar,
        BackendType.cpu_simd,
        BackendType.gpu_webgpu,
        BackendType.gpu_cuda,
    }) |bt| {
        if (comptime !bt.isImplemented()) continue;
        const ns = timeAdd(T, bt, out, a, b, iters);
        if (ns < best_ns) {
            best_ns = ns;
            best = bt;
        }
    }
    return best orelse error.NoBackend;
}
