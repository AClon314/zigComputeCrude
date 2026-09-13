// Stream compaction part 2: move flagged elements to their scan offset.
// Elements are 4-byte words, so the same shader compacts u32 or f32 payloads
// (f32 bit patterns are preserved exactly).
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn compact_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = nwg.x * 64u;
    let index = gid.x + gid.y * row;
    if (index >= params.count) {
        return;
    }
    if (flags[index] != 0u) {
        output[offsets[index]] = values[index];
    }
}
