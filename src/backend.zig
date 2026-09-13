const std = @import("std");

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

    /// gpu_webgpu is a native wgpu-native implementation.  Its runtime
    /// initialization is still fallible; callers must retain the CPU fallback.
    pub fn isImplemented(self: BackendType) bool {
        return switch (self) {
            .cpu_scalar, .cpu_simd, .gpu_webgpu => true,
            .gpu_cuda => false,
        };
    }
};

pub const SelectionMode = enum { manual, heuristic, benchmark };

/// 启发式：逐元素内核在 N 足够大时 SIMD 更有优势。
pub const simd_threshold: usize = 1024;
pub fn heuristic(size: usize) BackendType {
    return if (size >= simd_threshold) .cpu_simd else .cpu_scalar;
}

test "backend enum basics" {
    try std.testing.expectEqualStrings("cpu_simd", BackendType.cpu_simd.name());
    try std.testing.expect(BackendType.cpu_scalar.isImplemented());
    try std.testing.expect(BackendType.gpu_webgpu.isImplemented());
    try std.testing.expect(!BackendType.gpu_cuda.isImplemented());
    try std.testing.expectEqual(BackendType.cpu_simd, heuristic(4096));
    try std.testing.expectEqual(BackendType.cpu_scalar, heuristic(128));
}
