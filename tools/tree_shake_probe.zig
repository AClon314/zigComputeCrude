//! Tree-shake probe: a consumer that imports the published module but only uses
//! the CPU path.  `tools/check_tree_shake.sh` asserts the emitted object does
//! not contain GPU (`wgpu*`) or spatial (`bvh`/`grid_hash`) code.
//!
//! This is the build-time guarantee behind the monorepo layout: unused
//! declarations are never semantically analyzed, so unused modules/kernels cost
//! nothing to consumers.

const accel = @import("computeAccel");

export fn treeShakeCpuOnly(n: usize, out: [*]f32, a: [*]const f32, b: [*]const f32) void {
    accel.ComputeEngine(.cpu_simd).add(f32, out[0..n], a[0..n], b[0..n]);
}
