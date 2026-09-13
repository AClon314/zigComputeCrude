// Test-only shader for `Chain.dispatchIndirect` (part 2): stamp `params.value`
// over the range covered by the (indirectly provided) workgroup count.
struct Params {
    value: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn fill(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block = wid.x + wid.y * nwg.x;
    let index = block * 64u + lid;
    if (index >= arrayLength(&data)) {
        return;
    }
    data[index] = params.value;
}
