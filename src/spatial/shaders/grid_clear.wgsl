// Grid build pass 1: zero the per-cell counters.
struct Params {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    cell_size: f32,
    gx: u32,
    gy: u32,
    gz: u32,
    point_count: u32,
    query_count: u32,
    radius: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> counts: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn clear(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = nwg.x * 64u;
    let index = gid.x + gid.y * row;
    if (index >= arrayLength(&counts)) {
        return;
    }
    counts[index] = 0u;
}
