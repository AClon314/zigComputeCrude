// Test-only shader for `Chain.dispatchIndirect` (part 1): write the workgroup
// count of the next dispatch into a control buffer.
//
// Note: the control buffer must NOT be bound as storage in the dispatch that
// consumes it as `indirect` — wgpu treats storage access as an exclusive usage
// within a dispatch's usage scope.  Hence the split into two shaders.
struct Params {
    limit: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> control: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(1)
fn set_count() {
    control[0] = min(params.limit, 64u);
    control[1] = 1u;
    control[2] = 1u;
}
