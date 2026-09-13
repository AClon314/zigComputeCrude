const std = @import("std");

/// Minimal consumer of the `computeAccel` package:
///   * imports only the `computeAccel` module (never `computeAccel_spatial`);
///   * asks for a CPU-only build (`-Dwebgpu=false`), so wgpu-native is neither
///     linked nor required to be present.
/// `tools/check_consumer.sh` builds this and asserts the binary has no wgpu.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const accel = b.dependency("computeAccel", .{
        .target = target,
        .optimize = optimize,
        .webgpu = false,
    });

    const exe = b.addExecutable(.{
        .name = "cpu_consumer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "computeAccel", .module = accel.module("computeAccel") },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    b.step("run", "Build and run the CPU-only consumer").dependOn(&run_cmd.step);
}
