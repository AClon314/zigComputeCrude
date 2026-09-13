// GEMM baseline: one invocation per output element, global memory only.
//
//   C[row, col] = sum_k A[row, k] * B[k, col]     (row-major A, B, C)
//
// This is deliberately the "slow but obviously correct" variant from the
// Step 3 roadmap: no workgroup memory and no barriers, so it can be compared
// element-by-element against the CPU reference and against the tiled kernel.
// Every A/B element is re-read once per output element, which is exactly the
// memory-bound pattern the tiled variant improves on.
struct Params {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Recover the linear element index from the flattened 2D grid, matching
    // add.wgsl / saxpy.wgsl so one dispatch can exceed 65535 workgroups.
    let row_width = num_workgroups.x * 64u;
    let index = global_id.x + global_id.y * row_width;
    let total = params.m * params.n;
    if (index >= total) {
        return;
    }

    let row = index / params.n;
    let col = index % params.n;
    var acc: f32 = 0.0;
    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        acc = acc + a[row * params.k + kk] * b[kk * params.n + col];
    }
    c[index] = acc;
}
