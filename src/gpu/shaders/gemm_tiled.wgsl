// GEMM with workgroup (shared) memory tiling.
//
// One workgroup computes a TILE_M x TILE_N output tile.  For every TILE_K
// slice of the reduction dimension the 64 invocations cooperatively copy the
// needed A/B blocks into workgroup memory, sync, and then each invocation
// accumulates a 2x2 micro-tile out of that shared block.  Each A/B element is
// therefore read from global memory once per tile row/column instead of once
// per output element, which is what turns GEMM from memory-bound into
// compute-bound.
const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

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

var<workgroup> tile_a: array<f32, TILE_M * TILE_K>;
var<workgroup> tile_b: array<f32, TILE_K * TILE_N>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let row0 = workgroup_id.y * TILE_M;
    let col0 = workgroup_id.x * TILE_N;

    // 64 invocations x 2x2 micro-tile = 16x16 outputs.
    let micro_row = (local_index / 8u) * 2u;
    let micro_col = (local_index % 8u) * 2u;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    let k_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var kt: u32 = 0u; kt < k_tiles; kt = kt + 1u) {
        let k0 = kt * TILE_K;

        // 256 elements per tile, 4 per invocation.  Out-of-range edges are
        // zero-filled so the accumulation stays correct for arbitrary m/k/n.
        for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
            let flat = local_index + lane * 64u;

            let row = flat / TILE_K;
            let col_k = flat % TILE_K;
            var value_a: f32 = 0.0;
            if (row0 + row < params.m && k0 + col_k < params.k) {
                value_a = a[(row0 + row) * params.k + (k0 + col_k)];
            }
            tile_a[flat] = value_a;

            let row_k = flat / TILE_N;
            let col = flat % TILE_N;
            var value_b: f32 = 0.0;
            if (k0 + row_k < params.k && col0 + col < params.n) {
                value_b = b[(k0 + row_k) * params.n + (col0 + col)];
            }
            tile_b[flat] = value_b;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
            let a0 = tile_a[(micro_row + 0u) * TILE_K + kk];
            let a1 = tile_a[(micro_row + 1u) * TILE_K + kk];
            let b0 = tile_b[kk * TILE_N + (micro_col + 0u)];
            let b1 = tile_b[kk * TILE_N + (micro_col + 1u)];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
    }

    let out_row = row0 + micro_row;
    let out_col = col0 + micro_col;
    if (out_row < params.m && out_col < params.n) {
        c[out_row * params.n + out_col] = acc00;
    }
    if (out_row < params.m && out_col + 1u < params.n) {
        c[out_row * params.n + (out_col + 1u)] = acc01;
    }
    if (out_row + 1u < params.m && out_col < params.n) {
        c[(out_row + 1u) * params.n + out_col] = acc10;
    }
    if (out_row + 1u < params.m && out_col + 1u < params.n) {
        c[(out_row + 1u) * params.n + (out_col + 1u)] = acc11;
    }
}
