// Grid build pass 4: cursor = offsets (the scatter cursor starts at each cell's
// first slot).  A separate kernel keeps the offsets intact for queries.
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

@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> cursor: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn copy_offsets(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = nwg.x * 64u;
    let index = gid.x + gid.y * row;
    if (index >= arrayLength(&offsets)) {
        return;
    }
    cursor[index] = offsets[index];
}
