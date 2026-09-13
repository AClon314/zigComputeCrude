// Grid build pass 5: scatter point indices into their cell's slot range using
// an atomic cursor (same iteration order as grid_count).
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

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> slots: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn cellIndex(p: vec3<f32>) -> i32 {
    if (p.x < params.min_x || p.y < params.min_y || p.z < params.min_z) {
        return -1;
    }
    let fx = (p.x - params.min_x) / params.cell_size;
    let fy = (p.y - params.min_y) / params.cell_size;
    let fz = (p.z - params.min_z) / params.cell_size;
    if (fx >= f32(params.gx) || fy >= f32(params.gy) || fz >= f32(params.gz)) {
        return -1;
    }
    let cx = u32(floor(fx));
    let cy = u32(floor(fy));
    let cz = u32(floor(fz));
    return i32((cz * params.gy + cy) * params.gx + cx);
}

@compute @workgroup_size(64)
fn scatter_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = nwg.x * 64u;
    let index = gid.x + gid.y * row;
    if (index >= params.point_count) {
        return;
    }
    let cell = cellIndex(points[index].xyz);
    if (cell < 0) {
        return;
    }
    let slot = atomicAdd(&cursor[u32(cell)], 1u);
    slots[slot] = index;
}
