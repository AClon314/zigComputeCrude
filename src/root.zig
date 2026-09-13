const std = @import("std");
const Io = std.Io;
const backend = @import("backend.zig");
const buffer = @import("buffer.zig");
const engine = @import("engine.zig");
const bench_mod = @import("bench.zig");

pub const BackendType = backend.BackendType;
pub const SelectionMode = backend.SelectionMode;
pub const DeviceBuffer = buffer.DeviceBuffer;
pub const ComputeEngine = engine.ComputeEngine;
pub const bench = bench_mod;

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

// 其它 test 块分散在 submodule，root 引用即可。
