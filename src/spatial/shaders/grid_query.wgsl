// Radius query: count indexed points within `radius` of each query point.
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
    max_neighbors: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> neighbors: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

fn insideGrid(p: vec3<f32>) -> bool {
    if (p.x < params.min_x || p.y < params.min_y || p.z < params.min_z) {
        return false;
    }
    return (p.x - params.min_x) / params.cell_size < f32(params.gx)
        && (p.y - params.min_y) / params.cell_size < f32(params.gy)
        && (p.z - params.min_z) / params.cell_size < f32(params.gz);
}

fn clampCell(value: f32, origin: f32, dim: u32) -> i32 {
    let f = (value - origin) / params.cell_size;
    if (f <= 0.0) {
        return 0;
    }
    return min(i32(floor(f)), i32(dim) - 1);
}

@compute @workgroup_size(64)
fn query_counts(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = nwg.x * 64u;
    let q = gid.x + gid.y * row;
    if (q >= params.query_count) {
        return;
    }

    let center = queries[q].xyz;
    var total: u32 = 0u;
    if (insideGrid(center)) {
        let radius = params.radius;
        let x0 = clampCell(center.x - radius, params.min_x, params.gx);
        let x1 = clampCell(center.x + radius, params.min_x, params.gx);
        let y0 = clampCell(center.y - radius, params.min_y, params.gy);
        let y1 = clampCell(center.y + radius, params.min_y, params.gy);
        let z0 = clampCell(center.z - radius, params.min_z, params.gz);
        let z1 = clampCell(center.z + radius, params.min_z, params.gz);

        let r2 = radius * radius;
        var iz = z0;
        loop {
            if (iz > z1) { break; }
            var iy = y0;
            loop {
                if (iy > y1) { break; }
                var ix = x0;
                loop {
                    if (ix > x1) { break; }
                    let cell = u32((u32(iz) * params.gy + u32(iy)) * params.gx + u32(ix));
                    let start = offsets[cell];
                    let end = start + counts[cell];
                    var slot = start;
                    loop {
                        if (slot >= end) { break; }
                        let other = slots[slot];
                        let d = points[other].xyz - center;
                        if (dot(d, d) <= r2) {
                            if (total < params.max_neighbors) {
                                neighbors[q * params.max_neighbors + total] = other;
                            }
                            total = total + 1u;
                        }
                        slot = slot + 1u;
                    }
                    ix = ix + 1;
                }
                iy = iy + 1;
            }
            iz = iz + 1;
        }
    }
    out_counts[q] = total;
}
