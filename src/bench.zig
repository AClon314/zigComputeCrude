const std = @import("std");
const backend = @import("backend.zig");
const engine = @import("engine.zig");
const gpu_context = @import("gpu/context.zig");
const gpu_pipeline = @import("gpu/pipeline.zig");
const BackendType = backend.BackendType;
const ComputeEngine = engine.ComputeEngine;

/// 单调时钟（nanosecond），Zig 0.16 用 posix.clock_gettime。
pub fn nowNs() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// 对固定 CPU 后端跑 iters 次 add，返回总耗时（ns）。GPU 不经过此函数，
/// 否则 GPU 错误可能被 ComputeEngine 的 CPU fallback 隐藏。
pub fn timeAdd(
    comptime T: type,
    comptime bt: BackendType,
    out: []T,
    a: []const T,
    b: []const T,
    iters: usize,
) u64 {
    const t0 = nowNs();
    for (0..iters) |_| ComputeEngine(bt).add(T, out, a, b);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// 端到端 GPU 测量：每次迭代都走上传、dispatch、copy、map/readback。
/// 这里直接调用返回 error 的 GPU API，绝不把失败算成 CPU 时间。
pub fn timeGpuAdd(out: []f32, a: []const f32, b: []const f32, iters: usize) !u64 {
    const t0 = nowNs();
    for (0..iters) |_| try gpu_pipeline.add(out, a, b);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// Steady-state comparison: the same inputs are uploaded once, then `iters`
/// dispatches are submitted before one staging readback.  This is deliberately
/// separate from `timeGpuAdd` and is never used by backend selection.
pub fn timeGpuAddBatched(out: []f32, a: []const f32, b: []const f32, iters: usize) !u64 {
    const t0 = nowNs();
    try gpu_pipeline.addBatched(out, a, b, iters);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// Measurement details retained for the CLI.  In particular, `gpu_ns` is an
/// end-to-end number only; a missing value means the GPU was not a candidate or
/// its real operation failed.  `timeGpuAddBatched` is intentionally absent.
pub const PickReport = struct {
    selected: BackendType,
    scalar_ns: u64,
    simd_ns: u64,
    gpu_ns: ?u64,
    gpu_probe: gpu_context.ProbeResult,
    gpu_failure_reason: ?[]const u8,
};

var last_report: ?PickReport = null;

pub fn lastPickReport() ?PickReport {
    return last_report;
}

fn gpuRequestCanRun(size: usize, gpu_probe: gpu_context.ProbeResult) bool {
    if (size > std.math.maxInt(usize) / @sizeOf(f32)) return false;
    const groups = size / 64 +
        (if (size % 64 == 0) @as(usize, 0) else @as(usize, 1));
    return gpu_probe.canRun(size * @sizeOf(f32), groups);
}

/// Run the same selection algorithm with an explicit probe result.  Production
/// calls use `GpuContext.probe()`; the parameter also makes the no-GPU path
/// deterministic and testable without manufacturing a fake WebGPU device.
pub fn pickBestWithProbe(
    allocator: std.mem.Allocator,
    comptime T: type,
    size: usize,
    iters: usize,
    gpu_probe: gpu_context.ProbeResult,
) !BackendType {
    const report = try benchmarkWithProbe(allocator, T, size, iters, gpu_probe);
    last_report = report;
    return report.selected;
}

pub fn benchmarkWithProbe(
    allocator: std.mem.Allocator,
    comptime T: type,
    size: usize,
    iters: usize,
    gpu_probe: gpu_context.ProbeResult,
) !PickReport {
    const a = try allocator.alloc(T, size);
    defer allocator.free(a);
    const b = try allocator.alloc(T, size);
    defer allocator.free(b);
    const out = try allocator.alloc(T, size);
    defer allocator.free(out);
    @memset(a, 2.0);
    @memset(b, 3.0);

    const scalar_ns = timeAdd(T, .cpu_scalar, out, a, b, iters);
    const simd_ns = timeAdd(T, .cpu_simd, out, a, b, iters);

    var selected: BackendType = .cpu_scalar;
    var best_ns = scalar_ns;
    if (simd_ns < best_ns) {
        best_ns = simd_ns;
        selected = .cpu_simd;
    }

    var gpu_ns: ?u64 = null;
    var gpu_failure_reason: ?[]const u8 = null;
    if (comptime T == f32) {
        if (gpu_probe.available and gpuRequestCanRun(size, gpu_probe)) {
            gpu_ns = timeGpuAdd(out, a, b, iters) catch |err| blk: {
                gpu_failure_reason = gpu_pipeline.lastFallbackReason() orelse @errorName(err);
                break :blk null;
            };
            if (gpu_ns) |measured_ns| {
                if (measured_ns < best_ns) {
                    best_ns = measured_ns;
                    selected = .gpu_webgpu;
                }
            }
        } else if (!gpu_probe.available) {
            gpu_failure_reason = gpu_probe.reason;
        } else {
            gpu_failure_reason = gpu_context.ProbeFailure.request_exceeds_limits.reason();
        }
    } else {
        gpu_failure_reason = gpu_context.ProbeFailure.unsupported_type.reason();
    }

    return .{
        .selected = selected,
        .scalar_ns = scalar_ns,
        .simd_ns = simd_ns,
        .gpu_ns = gpu_ns,
        .gpu_probe = gpu_probe,
        .gpu_failure_reason = gpu_failure_reason,
    };
}

/// 运行时 bench 每个可用后端，返回最快的。GPU 只有 probe 成功且其
/// 端到端调用成功时才会进入比较；不会把 GPU fallback 的 CPU 时间算进去。
pub fn pickBest(allocator: std.mem.Allocator, comptime T: type, size: usize, iters: usize) !BackendType {
    const gpu_probe = if (comptime T == f32)
        gpu_context.GpuContext.probe()
    else
        gpu_context.ProbeResult.unavailable(.unsupported_type);
    return pickBestWithProbe(allocator, T, size, iters, gpu_probe);
}

test "pickBest excludes an unavailable GPU probe" {
    const selected = try pickBestWithProbe(
        std.testing.allocator,
        f32,
        128,
        1,
        gpu_context.ProbeResult.unavailable(.adapter_unavailable),
    );
    try std.testing.expect(selected != .gpu_webgpu);

    const report = lastPickReport().?;
    try std.testing.expect(!report.gpu_probe.available);
    try std.testing.expectEqualStrings(
        "WebGPU adapter request failed or returned no adapter",
        report.gpu_failure_reason.?,
    );
}

test "pickBest excludes an available probe that cannot run the request" {
    const report = try benchmarkWithProbe(
        std.testing.allocator,
        f32,
        128,
        1,
        .{
            .available = true,
            .failure = .none,
            .reason = gpu_context.ProbeFailure.none.reason(),
            .limits = .{
                .maxComputeWorkgroupsPerDimension = 65_535,
                .maxStorageBufferBindingSize = 511,
                .maxBufferSize = 1 << 20,
            },
        },
    );
    try std.testing.expect(report.selected != .gpu_webgpu);
    try std.testing.expect(report.gpu_ns == null);
    try std.testing.expectEqualStrings(
        "requested GPU size exceeds WebGPU limits",
        report.gpu_failure_reason.?,
    );
}
