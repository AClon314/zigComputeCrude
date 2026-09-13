// Stream compaction part 3: write the compacted length and make the buffer
// immediately usable as a `dispatchIndirect` argument (x = count, y = 1, z = 1).
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> flags: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn compact_total() {
    var total: u32 = 0u;
    if (params.count > 0u) {
        let last = params.count - 1u;
        total = offsets[last] + select(0u, 1u, flags[last] != 0u);
    }
    count[0] = total;
    count[1] = 1u;
    count[2] = 1u;
}
