//! Uniform-grid spatial index (S1, CPU reference implementation).
//!
//! This is the *reference* the GPU build path is compared against
//! (docs/node-system-migration.md, milestone S1 "空间原语，按需").  It uses the
//! same data layout as the planned GPU kernels so results are element-wise
//! comparable:
//!
//! ```text
//! counts[cell]                      (GPU: atomicAdd per point)
//!   -> exclusive prefix sum
//! offsets[cell]                     (GPU: scan kernel)
//!   -> scatter with a cursor copy      (GPU: atomicAdd on cursor)
//! slots[offsets[cell] .. +counts]   (point indices, grouped by cell)
//! ```
//!
//! Everything is caller-allocated (no allocator, no hidden state), which keeps
//! the reference deterministic and lets the GPU buffers use the same sizes.

const std = @import("std");

pub const Vec3 = struct { x: f32, y: f32, z: f32 };

pub const Grid = struct {
    min: Vec3,
    cell_size: f32,
    gx: u32,
    gy: u32,
    gz: u32,

    pub fn cellCount(self: Grid) usize {
        return @as(usize, self.gx) * @as(usize, self.gy) * @as(usize, self.gz);
    }

    /// Cell of a position, or null when the position lies outside the grid.
    /// "Outside" is `p < min` or `p >= min + cell_size * dim` on any axis
    /// (matching `insideGrid`/`cellIndex` in the GPU kernels);
    /// `gridCovering` pads the extent so every data point stays inside.
    pub fn cellOf(self: Grid, p: Vec3) ?[3]u32 {
        if (self.gx == 0 or self.gy == 0 or self.gz == 0) return null;
        if (p.x < self.min.x or p.y < self.min.y or p.z < self.min.z) return null;

        const fx = (p.x - self.min.x) / self.cell_size;
        const fy = (p.y - self.min.y) / self.cell_size;
        const fz = (p.z - self.min.z) / self.cell_size;
        if (fx < 0 or fy < 0 or fz < 0) return null;
        if (fx >= @as(f32, @floatFromInt(self.gx)) or
            fy >= @as(f32, @floatFromInt(self.gy)) or
            fz >= @as(f32, @floatFromInt(self.gz)))
        {
            return null;
        }

        const cx: u32 = @intFromFloat(@floor(fx));
        const cy: u32 = @intFromFloat(@floor(fy));
        const cz: u32 = @intFromFloat(@floor(fz));
        return .{ cx, cy, cz };
    }

    pub fn cellId(self: Grid, cell: [3]u32) u32 {
        return (cell[2] * self.gy + cell[1]) * self.gx + cell[0];
    }
};

/// Build the index.
///
/// `counts`, `offsets`, `cursor` are `grid.cellCount()` long; `slots` is
/// `points.len` long.  `cursor` is scratch (same role as the atomic cursor in
/// the GPU kernel); it may alias no other input.
pub fn build(
    grid: Grid,
    xs: []const f32,
    ys: []const f32,
    zs: []const f32,
    counts: []u32,
    offsets: []u32,
    cursor: []u32,
    slots: []u32,
) void {
    std.debug.assert(xs.len == ys.len and ys.len == zs.len);
    std.debug.assert(counts.len == grid.cellCount());
    std.debug.assert(offsets.len == counts.len and cursor.len == counts.len);
    std.debug.assert(slots.len == xs.len);

    @memset(counts, 0);
    for (0..xs.len) |i| {
        const cell = grid.cellOf(.{ .x = xs[i], .y = ys[i], .z = zs[i] }) orelse continue;
        counts[grid.cellId(cell)] += 1;
    }

    var running: u32 = 0;
    for (counts, 0..) |count, cell| {
        offsets[cell] = running;
        cursor[cell] = running;
        running += count;
    }

    for (0..xs.len) |i| {
        const cell = grid.cellOf(.{ .x = xs[i], .y = ys[i], .z = zs[i] }) orelse continue;
        const id = grid.cellId(cell);
        slots[cursor[id]] = @intCast(i);
        cursor[id] += 1;
    }
}

fn clampCell(value: f32, origin: f32, cell_size: f32, dim: u32) u32 {
    const f = (value - origin) / cell_size;
    if (f <= 0) return 0;
    const index: u32 = @intFromFloat(@floor(f));
    return @min(index, dim - 1);
}

/// Number of indexed points within `radius` of every query point.
///
/// The cell range is clamped to the grid, so centers whose sphere sticks out of
/// the grid may under-count points that were never indexed (same contract as
/// the GPU query kernel).  Points outside the grid are not indexed.
pub fn queryCounts(
    grid: Grid,
    xs: []const f32,
    ys: []const f32,
    zs: []const f32,
    counts: []const u32,
    offsets: []const u32,
    slots: []const u32,
    qx: []const f32,
    qy: []const f32,
    qz: []const f32,
    radius: f32,
    out: []u32,
) void {
    std.debug.assert(out.len == qx.len and qy.len == qx.len and qz.len == qx.len);
    const r2 = radius * radius;

    for (0..qx.len) |q| {
        const center = Vec3{ .x = qx[q], .y = qy[q], .z = qz[q] };
        var total: u32 = 0;
        if (grid.cellOf(center) != null) {
            // Cell range of the query sphere's AABB (clamped): using the center
            // cell +- floor(radius/cell) would under-count points near cell
            // edges, so derive the range from the actual bounds.
            const ix0 = clampCell(center.x - radius, grid.min.x, grid.cell_size, grid.gx);
            const ix1 = clampCell(center.x + radius, grid.min.x, grid.cell_size, grid.gx);
            const iy0 = clampCell(center.y - radius, grid.min.y, grid.cell_size, grid.gy);
            const iy1 = clampCell(center.y + radius, grid.min.y, grid.cell_size, grid.gy);
            const iz0 = clampCell(center.z - radius, grid.min.z, grid.cell_size, grid.gz);
            const iz1 = clampCell(center.z + radius, grid.min.z, grid.cell_size, grid.gz);

            var iz = iz0;
            while (iz <= iz1) : (iz += 1) {
                var iy = iy0;
                while (iy <= iy1) : (iy += 1) {
                    var ix = ix0;
                    while (ix <= ix1) : (ix += 1) {
                        const id = grid.cellId(.{ ix, iy, iz });
                        const start = offsets[id];
                        const end = start + counts[id];
                        for (slots[start..end]) |point| {
                            const dx = xs[point] - center.x;
                            const dy = ys[point] - center.y;
                            const dz = zs[point] - center.z;
                            if (dx * dx + dy * dy + dz * dz <= r2) total += 1;
                        }
                    }
                }
            }
        }
        out[q] = total;
    }
}

/// Neighbor lists: like `queryCounts`, but also stores up to `max_neighbors`
/// point indices per query at `out_neighbors[q * max_neighbors + j]` (slot
/// order, which is nondeterministic on the GPU; compare sorted in tests).
pub fn queryNeighbors(
    grid: Grid,
    xs: []const f32,
    ys: []const f32,
    zs: []const f32,
    counts: []const u32,
    offsets: []const u32,
    slots: []const u32,
    qx: []const f32,
    qy: []const f32,
    qz: []const f32,
    radius: f32,
    max_neighbors: usize,
    out_counts: []u32,
    out_neighbors: []u32,
) void {
    std.debug.assert(out_counts.len == qx.len);
    std.debug.assert(out_neighbors.len >= qx.len * max_neighbors);
    const r2 = radius * radius;

    for (0..qx.len) |q| {
        const center = Vec3{ .x = qx[q], .y = qy[q], .z = qz[q] };
        var total: u32 = 0;
        if (grid.cellOf(center) != null) {
            const ix0 = clampCell(center.x - radius, grid.min.x, grid.cell_size, grid.gx);
            const ix1 = clampCell(center.x + radius, grid.min.x, grid.cell_size, grid.gx);
            const iy0 = clampCell(center.y - radius, grid.min.y, grid.cell_size, grid.gy);
            const iy1 = clampCell(center.y + radius, grid.min.y, grid.cell_size, grid.gy);
            const iz0 = clampCell(center.z - radius, grid.min.z, grid.cell_size, grid.gz);
            const iz1 = clampCell(center.z + radius, grid.min.z, grid.cell_size, grid.gz);

            var iz = iz0;
            while (iz <= iz1) : (iz += 1) {
                var iy = iy0;
                while (iy <= iy1) : (iy += 1) {
                    var ix = ix0;
                    while (ix <= ix1) : (ix += 1) {
                        const id = grid.cellId(.{ ix, iy, iz });
                        const start = offsets[id];
                        const end = start + counts[id];
                        for (slots[start..end]) |point| {
                            const dx = xs[point] - center.x;
                            const dy = ys[point] - center.y;
                            const dz = zs[point] - center.z;
                            if (dx * dx + dy * dy + dz * dz <= r2) {
                                if (total < max_neighbors) {
                                    out_neighbors[q * max_neighbors + total] = point;
                                }
                                total += 1;
                            }
                        }
                    }
                }
            }
        }
        out_counts[q] = total;
    }
}

/// O(points x queries) reference, used by tests to validate `queryCounts`
/// (it counts every point, including ones the grid may not index).
pub fn bruteForceCounts(
    xs: []const f32,
    ys: []const f32,
    zs: []const f32,
    qx: []const f32,
    qy: []const f32,
    qz: []const f32,
    radius: f32,
    out: []u32,
) void {
    const r2 = radius * radius;
    for (0..qx.len) |q| {
        var total: u32 = 0;
        for (0..xs.len) |i| {
            const dx = xs[i] - qx[q];
            const dy = ys[i] - qy[q];
            const dz = zs[i] - qz[q];
            if (dx * dx + dy * dy + dz * dz <= r2) total += 1;
        }
        out[q] = total;
    }
}

/// Grid that covers every point (uniform cubic cells, `dim` per axis).
pub fn gridCovering(xs: []const f32, ys: []const f32, zs: []const f32, dim: u32) Grid {
    var lo = Vec3{ .x = std.math.inf(f32), .y = std.math.inf(f32), .z = std.math.inf(f32) };
    var hi = Vec3{ .x = -std.math.inf(f32), .y = -std.math.inf(f32), .z = -std.math.inf(f32) };
    for (0..xs.len) |i| {
        lo.x = @min(lo.x, xs[i]);
        lo.y = @min(lo.y, ys[i]);
        lo.z = @min(lo.z, zs[i]);
        hi.x = @max(hi.x, xs[i]);
        hi.y = @max(hi.y, ys[i]);
        hi.z = @max(hi.z, zs[i]);
    }
    const extent = @max(hi.x - lo.x, @max(hi.y - lo.y, hi.z - lo.z));
    // 1.001 padding keeps the maximum coordinate strictly inside the last cell.
    const cell_size = if (extent > 0) (extent / @as(f32, @floatFromInt(dim))) * 1.001 else 1.0;
    return .{ .min = lo, .cell_size = cell_size, .gx = dim, .gy = dim, .gz = dim };
}

// ---- tests ----

const TestData = struct {
    xs: []f32,
    ys: []f32,
    zs: []f32,
};

fn fillRandom(points: *TestData, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (points.xs, points.ys, points.zs) |*x, *y, *z| {
        x.* = random.float(f32) * 10.0;
        y.* = random.float(f32) * 10.0;
        z.* = random.float(f32) * 10.0;
    }
}

test "grid build indexes every in-grid point exactly once" {
    const gpa = std.testing.allocator;
    const n: usize = 4096;
    var points = TestData{
        .xs = try gpa.alloc(f32, n),
        .ys = try gpa.alloc(f32, n),
        .zs = try gpa.alloc(f32, n),
    };
    defer gpa.free(points.xs);
    defer gpa.free(points.ys);
    defer gpa.free(points.zs);
    fillRandom(&points, 7);

    const grid = gridCovering(points.xs, points.ys, points.zs, 16);
    const cells = grid.cellCount();
    const counts = try gpa.alloc(u32, cells);
    defer gpa.free(counts);
    const offsets = try gpa.alloc(u32, cells);
    defer gpa.free(offsets);
    const cursor = try gpa.alloc(u32, cells);
    defer gpa.free(cursor);
    const slots = try gpa.alloc(u32, n);
    defer gpa.free(slots);
    const seen = try gpa.alloc(bool, n);
    defer gpa.free(seen);

    build(grid, points.xs, points.ys, points.zs, counts, offsets, cursor, slots);

    var total: u32 = 0;
    for (counts) |count| total += count;
    try std.testing.expectEqual(@as(u32, @intCast(n)), total);

    @memset(seen, false);
    for (slots) |point| {
        try std.testing.expect(!seen[point]);
        seen[point] = true;
    }

    // offsets are the exclusive prefix sum of counts
    var running: u32 = 0;
    for (counts, offsets) |count, offset| {
        try std.testing.expectEqual(running, offset);
        running += count;
    }
}

test "grid query matches brute force" {
    const gpa = std.testing.allocator;
    const n: usize = 2000;
    const q: usize = 64;
    const xs = try gpa.alloc(f32, n);
    defer gpa.free(xs);
    const ys = try gpa.alloc(f32, n);
    defer gpa.free(ys);
    const zs = try gpa.alloc(f32, n);
    defer gpa.free(zs);
    const qx = try gpa.alloc(f32, q);
    defer gpa.free(qx);
    const qy = try gpa.alloc(f32, q);
    defer gpa.free(qy);
    const qz = try gpa.alloc(f32, q);
    defer gpa.free(qz);

    var points = TestData{ .xs = xs, .ys = ys, .zs = zs };
    fillRandom(&points, 11);
    // Query points near the middle so radius spheres stay inside the grid.
    var prng = std.Random.DefaultPrng.init(12);
    const random = prng.random();
    for (qx, qy, qz) |*x, *y, *z| {
        x.* = 3.0 + random.float(f32) * 4.0;
        y.* = 3.0 + random.float(f32) * 4.0;
        z.* = 3.0 + random.float(f32) * 4.0;
    }

    const grid = gridCovering(xs, ys, zs, 16);
    const cells = grid.cellCount();
    const counts = try gpa.alloc(u32, cells);
    defer gpa.free(counts);
    const offsets = try gpa.alloc(u32, cells);
    defer gpa.free(offsets);
    const cursor = try gpa.alloc(u32, cells);
    defer gpa.free(cursor);
    const slots = try gpa.alloc(u32, n);
    defer gpa.free(slots);
    const grid_counts = try gpa.alloc(u32, q);
    defer gpa.free(grid_counts);
    const brute_counts = try gpa.alloc(u32, q);
    defer gpa.free(brute_counts);

    build(grid, xs, ys, zs, counts, offsets, cursor, slots);
    const radius: f32 = 0.75;
    queryCounts(grid, xs, ys, zs, counts, offsets, slots, qx, qy, qz, radius, grid_counts);
    bruteForceCounts(xs, ys, zs, qx, qy, qz, radius, brute_counts);

    try std.testing.expectEqualSlices(u32, brute_counts, grid_counts);
}
