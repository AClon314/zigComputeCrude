// Broadcast bias add (a "mix"-style node): result[i] = x[i] + alpha * bias[i % n].
// Used by the M0 heterogeneous chain (GEMM -> bias -> reduce) to show a second,
// differently-shaped kernel inside one submitted chain.
struct Params {
    n: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let row_width = num_workgroups.x * 64u;
    let index = global_id.x + global_id.y * row_width;
    if (index >= arrayLength(&x)) {
        return;
    }
    result[index] = x[index] + params.alpha * bias[index % params.n];
}
