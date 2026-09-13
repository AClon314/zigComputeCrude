// Exclusive prefix sum over u32, recorded as three dispatches in one chain:
//
//   block_scan        per-workgroup inclusive scan -> exclusive output + block totals
//   block_scan_global one workgroup scans the block totals in place (exclusive offsets)
//   scan_apply        output[i] += block_offsets[i / 64]
//
// All three entry points share one binding layout:
//   0 input   (storage, read)
//   1 output  (storage, read_write)
//   2 partials(storage, read_write)  block totals, then block offsets in place
//   3 params  (uniform, read)
// so a caller can build exactly one bind group per (input, output) pair.
//
// This is the primitive the spatial index needs (counts -> offsets -> scatter)
// and the base for compaction/stream compaction.
const WG: u32 = 64u;

struct Params {
    count: u32,
    block_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch: array<u32, WG>;

fn blockOf(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.x + wid.y * nwg.x;
}

fn inclusiveScan(lid: u32) {
    var offset: u32 = 1u;
    loop {
        if (offset >= WG) {
            break;
        }
        var other: u32 = 0u;
        if (lid >= offset) {
            other = scratch[lid - offset];
        }
        workgroupBarrier();
        if (lid >= offset) {
            scratch[lid] = scratch[lid] + other;
        }
        workgroupBarrier();
        offset = offset * 2u;
    }
}

@compute @workgroup_size(64)
fn block_scan(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block = blockOf(wid, nwg);
    let index = block * WG + lid;

    var value: u32 = 0u;
    if (index < params.count) {
        value = input[index];
    }
    scratch[lid] = value;
    workgroupBarrier();
    inclusiveScan(lid);

    let inclusive = scratch[lid];
    if (index < params.count) {
        output[index] = inclusive - value; // exclusive scan
    }
    if (lid == WG - 1u) {
        partials[block] = inclusive; // block total
    }
}

@compute @workgroup_size(64)
fn block_scan_global(@builtin(local_invocation_index) lid: u32) {
    // Thread `lid` owns a *contiguous* chunk of blocks [start, end): the
    // workgroup scan of per-thread totals then gives the exclusive prefix for
    // that chunk, and each thread walks its chunk writing offsets in place
    // (one owner per index).  A strided (lid, lid+64, ...) split would break
    // the prefix order, because block 64 sorts after block 1, not after
    // thread 0's other blocks.
    let chunk = (params.block_count + WG - 1u) / WG;
    let start = lid * chunk;
    let end = min(start + chunk, params.block_count);

    var total: u32 = 0u;
    var i = start;
    loop {
        if (i >= end) {
            break;
        }
        total = total + partials[i];
        i = i + 1u;
    }
    scratch[lid] = total;
    workgroupBarrier();
    inclusiveScan(lid);

    var running = scratch[lid] - total;
    var j = start;
    loop {
        if (j >= end) {
            break;
        }
        let block_total = partials[j];
        partials[j] = running;
        running = running + block_total;
        j = j + 1u;
    }
}

@compute @workgroup_size(64)
fn scan_apply(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block = blockOf(wid, nwg);
    let index = block * WG + lid;
    if (index >= params.count) {
        return;
    }
    output[index] = output[index] + partials[block];
}
