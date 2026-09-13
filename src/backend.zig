const std = @import("std");
const gpu_context = @import("gpu/context.zig");

pub const BackendType = enum {
    cpu_scalar,
    cpu_simd,
    gpu_webgpu,
    gpu_cuda,

    pub fn name(self: BackendType) []const u8 {
        return switch (self) {
            .cpu_scalar => "cpu_scalar",
            .cpu_simd => "cpu_simd",
            .gpu_webgpu => "gpu_webgpu",
            .gpu_cuda => "gpu_cuda",
        };
    }

    /// gpu_webgpu is a native wgpu-native implementation. Its runtime
    /// initialization is still fallible; callers must retain the CPU fallback.
    pub fn isImplemented(self: BackendType) bool {
        return switch (self) {
            .cpu_scalar, .cpu_simd, .gpu_webgpu => true,
            .gpu_cuda => false,
        };
    }
};

pub const SelectionMode = enum { manual, heuristic, benchmark };

/// 逐元素 add 的 GPU 启发式闸门。
///
/// T1 实测（AMD Vega，Debug，size=1<<20）显示 GPU 端到端为 4.652 GB/s，
/// cpu_simd 为 14.676 GB/s；因此不能在常见的 1<<20 工作量上仅凭“有 GPU”
/// 就选择 GPU。这里保守地把 GPU 候选规模设为 1<<22；到达闸门后仍需
/// `GpuContext.probe()` 成功，真正的 `--auto` 选择还会再做端到端 bench。
pub const gpu_threshold: usize = 1 << 22;

/// 启发式中的 CPU 闸门：逐元素内核在 N 足够大时 SIMD 更有优势。
pub const simd_threshold: usize = 1024;

fn gpuRequestCanRun(size: usize, probe: gpu_context.ProbeResult) bool {
    if (size > std.math.maxInt(usize) / @sizeOf(f32)) return false;
    const groups = size / 64 +
        (if (size % 64 == 0) @as(usize, 0) else @as(usize, 1));
    return probe.canRun(size * @sizeOf(f32), groups);
}

/// 供旧调用方和纯阈值单元测试共享的选择函数。生产 heuristic 不使用
/// 这个只有 bool 的兼容入口，而是必须经过 `heuristicWithGpuProbe` 的
/// probe + limits 双重闸门。
pub fn heuristicWithGpuAvailability(size: usize, gpu_available: bool) BackendType {
    if (size < gpu_threshold) {
        return if (size >= simd_threshold) .cpu_simd else .cpu_scalar;
    }
    return if (gpu_available) .gpu_webgpu else .cpu_simd;
}

/// 启发式的实际选择入口：GPU 必须既通过能力探测，又能在设备 limits
/// 下容纳请求的 f32 buffer 和 2D workgroup grid。
pub fn heuristicWithGpuProbe(size: usize, probe: gpu_context.ProbeResult) BackendType {
    if (size < gpu_threshold) return heuristicWithGpuAvailability(size, false);
    return if (gpuRequestCanRun(size, probe)) .gpu_webgpu else .cpu_simd;
}

/// 按规模选择后端；到 GPU 闸门前不会触发运行时初始化。
pub fn heuristic(size: usize) BackendType {
    if (size < gpu_threshold) return heuristicWithGpuAvailability(size, false);
    return heuristicWithGpuProbe(size, gpu_context.GpuContext.probe());
}

test "backend enum basics" {
    try std.testing.expectEqualStrings("cpu_simd", BackendType.cpu_simd.name());
    try std.testing.expect(BackendType.cpu_scalar.isImplemented());
    try std.testing.expect(BackendType.gpu_webgpu.isImplemented());
    try std.testing.expect(!BackendType.gpu_cuda.isImplemented());
    try std.testing.expectEqual(BackendType.cpu_simd, heuristicWithGpuAvailability(4096, false));
    try std.testing.expectEqual(BackendType.cpu_scalar, heuristicWithGpuAvailability(128, true));
}

test "heuristic keeps GPU behind the measured-size gate" {
    try std.testing.expectEqual(
        BackendType.cpu_simd,
        heuristicWithGpuAvailability(gpu_threshold - 1, true),
    );
    try std.testing.expectEqual(
        BackendType.gpu_webgpu,
        heuristicWithGpuAvailability(gpu_threshold, true),
    );
    try std.testing.expectEqual(
        BackendType.cpu_simd,
        heuristicWithGpuAvailability(gpu_threshold, false),
    );
}

test "heuristic requires a runnable GPU limits probe" {
    const probe = gpu_context.ProbeResult{
        .available = true,
        .failure = .none,
        .reason = gpu_context.ProbeFailure.none.reason(),
        .limits = .{
            .maxComputeWorkgroupsPerDimension = 65_535,
            .maxStorageBufferBindingSize = 1 << 30,
            .maxBufferSize = 1 << 30,
        },
    };
    try std.testing.expectEqual(
        BackendType.gpu_webgpu,
        heuristicWithGpuProbe(1 << 22, probe),
    );

    var too_small = probe;
    too_small.limits.maxBufferSize = (1 << 22) * @sizeOf(f32) - 1;
    try std.testing.expectEqual(
        BackendType.cpu_simd,
        heuristicWithGpuProbe(1 << 22, too_small),
    );
}
