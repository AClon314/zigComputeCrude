const std = @import("std");
const Io = std.Io;
const backend = @import("backend.zig");
const buffer = @import("buffer.zig");
const engine = @import("engine.zig");
const bench_mod = @import("bench.zig");
const gpu_context = @import("gpu/context.zig");
const gpu_pipeline = @import("gpu/pipeline.zig");

pub const BackendType = backend.BackendType;
pub const SelectionMode = backend.SelectionMode;
pub const DeviceBuffer = buffer.DeviceBuffer;
pub const ComputeEngine = engine.ComputeEngine;
pub const bench = bench_mod;
pub const gpu = gpu_pipeline;
pub const GpuContext = gpu_context.GpuContext;
pub const GpuLimits = gpu_context.GpuLimits;
pub const ProbeFailure = gpu_context.ProbeFailure;
pub const ProbeResult = gpu_context.ProbeResult;

/// Compatibility helper retained for the Task 1 CLI template.
pub fn printAnotherMessage(writer: *Io.Writer) Io.Writer.Error!void {
    try writer.print("Run `zig build test` to run the tests.\n", .{});
}

/// 选择逻辑编排：manual / heuristic / benchmark。
pub fn selectBackend(
    allocator: std.mem.Allocator,
    mode: SelectionMode,
    manual_choice: BackendType,
    comptime T: type,
    size: usize,
    iters: usize,
) !BackendType {
    return switch (mode) {
        .manual => manual_choice,
        .heuristic => backend.heuristic(size),
        .benchmark => try bench_mod.pickBest(allocator, T, size, iters),
    };
}

// root.zig 中：下面的子文件被引用，它们的 test 块才被 zig build test 纳入。
test {
    _ = backend;
    _ = buffer;
    _ = engine;
    _ = bench_mod;
    _ = gpu_context;
    _ = gpu_pipeline;
}

// ===== 关键 test：性能对比（scalar vs simd）=====
test "perf: simd beats scalar on large arrays" {
    const builtin = @import("builtin");
    const size: usize = 1 << 20; // 1,048,576
    const iters: usize = 20;

    const gpa = std.testing.allocator;
    const a = try gpa.alloc(f32, size);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, size);
    defer gpa.free(b);
    const out_scalar = try gpa.alloc(f32, size);
    defer gpa.free(out_scalar);
    const out_simd = try gpa.alloc(f32, size);
    defer gpa.free(out_simd);

    @memset(a, 2.0);
    @memset(b, 3.0);

    // 正确性：两后端结果必须一致
    ComputeEngine(.cpu_scalar).add(f32, out_scalar, a, b);
    ComputeEngine(.cpu_simd).add(f32, out_simd, a, b);
    try std.testing.expectEqualSlices(f32, out_scalar, out_simd);
    try std.testing.expectEqual(@as(f32, 5.0), out_scalar[0]);
    try std.testing.expectEqual(@as(f32, 5.0), out_scalar[size - 1]);

    const t_scalar = bench_mod.timeAdd(f32, .cpu_scalar, out_scalar, a, b, iters);
    const t_simd = bench_mod.timeAdd(f32, .cpu_simd, out_simd, a, b, iters);
    const speedup = @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_simd));

    std.debug.print("\n[perf perf] size={} iters={} scalar={}ns simd={}ns speedup={d:.2}x\n", .{ size, iters, t_scalar, t_simd, speedup });

    // ReleaseFast 下 SIMD 必须显著更快；Debug 下只打印并保证结果正确。
    if (builtin.mode == .ReleaseFast) {
        try std.testing.expect(t_simd < t_scalar);
        try std.testing.expect(speedup > 1.0);
    }
}

test "gpu webgpu add matches CPU backends" {
    const n: usize = 1 << 20;
    const gpa = std.testing.allocator;

    var context = gpu_context.GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();

    const a = try gpa.alloc(f32, n);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n);
    defer gpa.free(b);
    const out_gpu = try gpa.alloc(f32, n);
    defer gpa.free(out_gpu);
    const out_scalar = try gpa.alloc(f32, n);
    defer gpa.free(out_scalar);
    const out_simd = try gpa.alloc(f32, n);
    defer gpa.free(out_simd);

    for (0..n) |i| {
        a[i] = @as(f32, @floatFromInt(i % 97)) * 0.25;
        b[i] = @as(f32, @floatFromInt(i % 53)) * 0.5;
    }
    ComputeEngine(.cpu_scalar).add(f32, out_scalar, a, b);
    ComputeEngine(.cpu_simd).add(f32, out_simd, a, b);
    try gpu_pipeline.addWithContext(&context, out_gpu, a, b);
    try std.testing.expectEqualSlices(f32, out_scalar, out_simd);
    try std.testing.expectEqualSlices(f32, out_scalar, out_gpu);
}

test "gpu webgpu add uses 2D dispatch past 65535 workgroups" {
    const n: usize = 1 << 22;
    const groups = (n + 63) / 64;
    try std.testing.expectEqual(@as(usize, 65_536), groups);

    const gpa = std.testing.allocator;
    var context = gpu_context.GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();

    try std.testing.expect(context.canRun(n * @sizeOf(f32), groups));

    const a = try gpa.alloc(f32, n);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n);
    defer gpa.free(b);
    const out_gpu = try gpa.alloc(f32, n);
    defer gpa.free(out_gpu);
    const out_scalar = try gpa.alloc(f32, n);
    defer gpa.free(out_scalar);
    const out_simd = try gpa.alloc(f32, n);
    defer gpa.free(out_simd);

    @memset(a, 2.0);
    @memset(b, 3.0);
    ComputeEngine(.cpu_scalar).add(f32, out_scalar, a, b);
    ComputeEngine(.cpu_simd).add(f32, out_simd, a, b);

    gpu_pipeline.clearFallbackReason();
    try gpu_pipeline.addWithContext(&context, out_gpu, a, b);
    try std.testing.expect(gpu_pipeline.lastFallbackReason() == null);
    try std.testing.expectEqualSlices(f32, out_scalar, out_simd);
    try std.testing.expectEqualSlices(f32, out_scalar, out_gpu);
}

test "gpu webgpu saxpy matches CPU backends" {
    const n: usize = 1 << 20;
    const gpa = std.testing.allocator;

    var context = gpu_context.GpuContext.init(gpa) catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };
    defer context.deinit();

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const y = try gpa.alloc(f32, n);
    defer gpa.free(y);
    const out_gpu = try gpa.alloc(f32, n);
    defer gpa.free(out_gpu);
    const out_scalar = try gpa.alloc(f32, n);
    defer gpa.free(out_scalar);
    const out_simd = try gpa.alloc(f32, n);
    defer gpa.free(out_simd);

    for (0..n) |i| {
        x[i] = @as(f32, @floatFromInt(i % 97)) * 0.25;
        y[i] = @as(f32, @floatFromInt(i % 53)) * 0.5;
    }
    const alpha: f32 = 1.75;
    ComputeEngine(.cpu_scalar).saxpy(f32, alpha, out_scalar, x, y);
    ComputeEngine(.cpu_simd).saxpy(f32, alpha, out_simd, x, y);
    try gpu_pipeline.saxpyWithContext(&context, alpha, out_gpu, x, y);
    try std.testing.expectEqualSlices(f32, out_scalar, out_simd);
    try std.testing.expectEqualSlices(f32, out_scalar, out_gpu);
}

test "gpu fallback keeps the CPU result and exposes its reason" {
    gpu_context.setProbeOverrideForTesting(
        gpu_context.ProbeResult.unavailable(.adapter_unavailable),
    );
    defer gpu_context.resetGlobal();

    const a = [_]f32{2.0} ** 17;
    const b = [_]f32{3.0} ** 17;
    var out: [17]f32 = undefined;
    ComputeEngine(.gpu_webgpu).add(f32, &out, &a, &b);

    for (out) |value| try std.testing.expectEqual(@as(f32, 5.0), value);
    try std.testing.expectEqualStrings(
        "WebGPU adapter request failed or returned no adapter",
        gpu_context.lastFallbackReason().?,
    );
}

// 其它 test 块分散在 submodule，root 引用即可。
