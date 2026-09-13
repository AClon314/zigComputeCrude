// Two-pass workgroup reduction (sum / max).
//
// Pass 1: `groups` workgroups each reduce a grid-strided slice of the input
//         into `output[workgroup_id.x]` (one partial per workgroup).
// Pass 2: the same kernel is dispatched with one workgroup over the partials
//         buffer, writing the final value to `output[0]`.
//
// Each invocation keeps a private accumulator, then the workgroup combines the
// 64 accumulators through workgroup memory (the "shared blackboard") with a
// barrier-separated tree.  This is the low-risk shared-memory step of the
// Step 3 roadmap and the pattern a tiled GEMM reduction would reuse.
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> scratch: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(64)
fn sum_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total_threads = num_workgroups.x * WORKGROUP_SIZE;
    var acc: f32 = 0.0;
    for (
        var index = workgroup_id.x * WORKGROUP_SIZE + local_index;
        index < params.n;
        index = index + total_threads
    ) {
        acc = acc + input[index];
    }
    scratch[local_index] = acc;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local_index < stride) {
            scratch[local_index] = scratch[local_index] + scratch[local_index + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_index == 0u) {
        output[workgroup_id.x] = scratch[0];
    }
}

@compute @workgroup_size(64)
fn max_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total_threads = num_workgroups.x * WORKGROUP_SIZE;
    // f32 most-negative finite value; every real input replaces it.
    var acc: f32 = -3.402823466e+38;
    for (
        var index = workgroup_id.x * WORKGROUP_SIZE + local_index;
        index < params.n;
        index = index + total_threads
    ) {
        acc = max(acc, input[index]);
    }
    scratch[local_index] = acc;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local_index < stride) {
            scratch[local_index] = max(scratch[local_index], scratch[local_index + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_index == 0u) {
        output[workgroup_id.x] = scratch[0];
    }
}
