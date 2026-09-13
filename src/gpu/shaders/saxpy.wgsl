struct Params {
    alpha: f32,
};

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> y: array<f32>;

@group(0) @binding(2)
var<storage, read_write> result: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let row_width = num_workgroups.x * 64u;
    let i = global_id.x + global_id.y * row_width;
    if (i >= arrayLength(&x)) {
        return;
    }
    result[i] = params.alpha * x[i] + y[i];
}
