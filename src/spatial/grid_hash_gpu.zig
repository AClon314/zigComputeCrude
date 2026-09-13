//! GPU uniform-grid spatial index (S1) built on the core runtime.
//!
//! Pipeline, all recorded into one `Chain` (one submit, one readback policy):
//!
//!   clear counts -> count points (atomicAdd) -> scan counts -> offsets
//!     -> copy offsets -> cursor -> scatter points (atomicAdd)
//!   query: for each query point, visit the cells of the radius AABB and test
//!          the slots stored in [offsets[cell], offsets[cell] + counts[cell]).
//!
//! Layout is AoS `vec4<f32>` for points/queries so the query kernel stays
//! within the 8 storage-buffer limit of the WebGPU baseline.
//!
//! Contract (same as the CPU reference `grid_hash`):
//!   * points outside `[min, min + cell_size*dims]` are not indexed;
//!   * query centers outside the grid count 0;
//!   * upper-boundary cell coordinates are clamped into the last cell.

const std = @import("std");
const core = @import("computeAccel");
const runtime = core.runtime;
const primitives = core.primitives;
const reference = @import("grid_hash.zig");

const clear_shader = @embedFile("shaders/grid_clear.wgsl");
const count_shader = @embedFile("shaders/grid_count.wgsl");
const copy_shader = @embedFile("shaders/grid_copy.wgsl");
const scatter_shader = @embedFile("shaders/grid_scatter.wgsl");
const query_shader = @embedFile("shaders/grid_query.wgsl");

/// x, y, z + one unused lane (AoS keeps the query kernel under 8 bindings).
pub const Point = [4]f32;

/// Must match the `Params` struct in src/spatial/shaders/*.wgsl (48 bytes).
pub const GpuParams = extern struct {
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
    pad0: u32 = 0,
    pad1: u32 = 0,
};

pub const GridIndex = struct {
    ctx: *runtime.Device,
    grid: reference.Grid,
    cells: usize,
    max_points: usize,
    max_queries: usize,

    counts: runtime.Buffer,
    offsets: runtime.Buffer,
    cursor: runtime.Buffer,
    slots: runtime.Buffer,
    points_buf: runtime.Buffer,
    queries_buf: runtime.Buffer,
    out_counts: runtime.Buffer,
    params_build: runtime.Buffer,
    params_query: runtime.Buffer,

    clear_kernel: runtime.Kernel,
    count_kernel: runtime.Kernel,
    copy_kernel: runtime.Kernel,
    scatter_kernel: runtime.Kernel,
    query_kernel: runtime.Kernel,
    scanner: primitives.Scanner,

    bind_clear: ?*anyopaque,
    bind_count: ?*anyopaque,
    bind_copy: ?*anyopaque,
    bind_scatter: ?*anyopaque,
    bind_query: ?*anyopaque,
    scan_binding: primitives.ScanBinding,

    pub fn init(
        ctx: *runtime.Device,
        grid: reference.Grid,
        max_points: usize,
        max_queries: usize,
    ) !GridIndex {
        const cells = grid.cellCount();
        if (cells == 0 or max_points == 0 or max_queries == 0) return error.GpuError;

        const cell_bytes = cells * @sizeOf(u32);
        const storage_cell = runtime.buffer.storage_rw;

        var self = GridIndex{
            .ctx = ctx,
            .grid = grid,
            .cells = cells,
            .max_points = max_points,
            .max_queries = max_queries,
            .counts = try runtime.Buffer.init(ctx, cell_bytes, storage_cell),
            .offsets = undefined,
            .cursor = undefined,
            .slots = undefined,
            .points_buf = undefined,
            .queries_buf = undefined,
            .out_counts = undefined,
            .params_build = undefined,
            .params_query = undefined,
            .clear_kernel = undefined,
            .count_kernel = undefined,
            .copy_kernel = undefined,
            .scatter_kernel = undefined,
            .query_kernel = undefined,
            .scanner = undefined,
            .bind_clear = null,
            .bind_count = null,
            .bind_copy = null,
            .bind_scatter = null,
            .bind_query = null,
            .scan_binding = undefined,
        };
        errdefer self.deinit();

        self.offsets = try runtime.Buffer.init(ctx, cell_bytes, storage_cell);
        self.cursor = try runtime.Buffer.init(ctx, cell_bytes, storage_cell);
        self.slots = try runtime.Buffer.init(ctx, max_points * @sizeOf(u32), storage_cell);
        self.points_buf = try runtime.Buffer.init(ctx, max_points * @sizeOf(Point), runtime.buffer.storage_r);
        self.queries_buf = try runtime.Buffer.init(ctx, max_queries * @sizeOf(Point), runtime.buffer.storage_r);
        self.out_counts = try runtime.Buffer.init(ctx, max_queries * @sizeOf(u32), storage_cell);
        self.params_build = try runtime.Buffer.init(ctx, @sizeOf(GpuParams), runtime.buffer.uniform);
        self.params_query = try runtime.Buffer.init(ctx, @sizeOf(GpuParams), runtime.buffer.uniform);

        self.clear_kernel = try runtime.Kernel.init(ctx, clear_shader, "clear", &.{
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);
        self.count_kernel = try runtime.Kernel.init(ctx, count_shader, "count_points", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);
        self.copy_kernel = try runtime.Kernel.init(ctx, copy_shader, "copy_offsets", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);
        self.scatter_kernel = try runtime.Kernel.init(ctx, scatter_shader, "scatter_points", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);
        self.query_kernel = try runtime.Kernel.init(ctx, query_shader, "query_counts", &.{
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .read },
            .{ .kind = .storage, .access = .write },
            .{ .kind = .uniform, .access = .read },
        }, 64);

        self.scanner = try primitives.Scanner.init(ctx, primitives.scanBlockCount(cells));

        self.bind_clear = try self.clear_kernel.createBindGroup(&.{ &self.counts, &self.params_build });
        errdefer runtime.releaseBindGroup(self.bind_clear.?);
        self.bind_count = try self.count_kernel.createBindGroup(&.{ &self.points_buf, &self.counts, &self.params_build });
        errdefer runtime.releaseBindGroup(self.bind_count.?);
        self.bind_copy = try self.copy_kernel.createBindGroup(&.{ &self.offsets, &self.cursor, &self.params_build });
        errdefer runtime.releaseBindGroup(self.bind_copy.?);
        self.bind_scatter = try self.scatter_kernel.createBindGroup(&.{ &self.points_buf, &self.cursor, &self.slots, &self.params_build });
        errdefer runtime.releaseBindGroup(self.bind_scatter.?);
        self.bind_query = try self.query_kernel.createBindGroup(&.{
            &self.points_buf,
            &self.counts,
            &self.offsets,
            &self.slots,
            &self.queries_buf,
            &self.out_counts,
            &self.params_query,
        });
        errdefer runtime.releaseBindGroup(self.bind_query.?);
        self.scan_binding = try self.scanner.bind(&self.counts, &self.offsets);

        try self.writeStaticParams();
        return self;
    }

    pub fn deinit(self: *GridIndex) void {
        if (self.bind_query) |handle| runtime.releaseBindGroup(handle);
        if (self.bind_scatter) |handle| runtime.releaseBindGroup(handle);
        if (self.bind_copy) |handle| runtime.releaseBindGroup(handle);
        if (self.bind_count) |handle| runtime.releaseBindGroup(handle);
        if (self.bind_clear) |handle| runtime.releaseBindGroup(handle);
        self.bind_query = null;
        self.bind_scatter = null;
        self.bind_copy = null;
        self.bind_count = null;
        self.bind_clear = null;

        self.scanner.deinit();
        self.query_kernel.deinit();
        self.scatter_kernel.deinit();
        self.copy_kernel.deinit();
        self.count_kernel.deinit();
        self.clear_kernel.deinit();

        self.params_query.deinit(self.ctx);
        self.params_build.deinit(self.ctx);
        self.out_counts.deinit(self.ctx);
        self.queries_buf.deinit(self.ctx);
        self.points_buf.deinit(self.ctx);
        self.slots.deinit(self.ctx);
        self.cursor.deinit(self.ctx);
        self.offsets.deinit(self.ctx);
        self.counts.deinit(self.ctx);
    }

    fn writeStaticParams(self: *GridIndex) !void {
        var params = GpuParams{
            .min_x = self.grid.min.x,
            .min_y = self.grid.min.y,
            .min_z = self.grid.min.z,
            .cell_size = self.grid.cell_size,
            .gx = self.grid.gx,
            .gy = self.grid.gy,
            .gz = self.grid.gz,
            .point_count = 0,
            .query_count = 0,
            .radius = 0,
        };
        self.params_build.markHostDirty();
        try self.params_build.toDevice(self.ctx, std.mem.asBytes(&params));
        self.params_query.markHostDirty();
        try self.params_query.toDevice(self.ctx, std.mem.asBytes(&params));
    }

    /// Upload `points` and record the five build dispatches into `chain`.
    ///
    /// Params and point data are uploaded immediately, so a GridIndex records
    /// at most one build (and one query batch) per chain.
    pub fn build(self: *GridIndex, chain: *runtime.Chain, points: []const Point, point_count: usize) !void {
        if (point_count == 0 or point_count > self.max_points or point_count > points.len) {
            return error.GpuError;
        }

        var params = GpuParams{
            .min_x = self.grid.min.x,
            .min_y = self.grid.min.y,
            .min_z = self.grid.min.z,
            .cell_size = self.grid.cell_size,
            .gx = self.grid.gx,
            .gy = self.grid.gy,
            .gz = self.grid.gz,
            .point_count = @intCast(point_count),
            .query_count = 0,
            .radius = 0,
        };
        self.params_build.markHostDirty();
        try self.params_build.toDevice(self.ctx, std.mem.asBytes(&params));
        self.points_buf.markHostDirty();
        try self.points_buf.toDevice(self.ctx, std.mem.sliceAsBytes(points[0..point_count]));

        const cell_grid = try self.clear_kernel.gridLinear(self.cells);
        const point_grid = try self.count_kernel.gridLinear(point_count);

        try chain.dispatch(&self.clear_kernel, self.bind_clear.?, &.{ &self.counts, &self.params_build }, cell_grid);
        try chain.dispatch(&self.count_kernel, self.bind_count.?, &.{ &self.points_buf, &self.counts, &self.params_build }, point_grid);
        try self.scanner.run(chain, &self.scan_binding, &self.counts, &self.offsets, self.cells);
        try chain.dispatch(&self.copy_kernel, self.bind_copy.?, &.{ &self.offsets, &self.cursor, &self.params_build }, cell_grid);
        try chain.dispatch(&self.scatter_kernel, self.bind_scatter.?, &.{ &self.points_buf, &self.cursor, &self.slots, &self.params_build }, point_grid);
    }

    /// Upload `queries` and record the radius-query dispatch into `chain`.
    /// Read `out_counts` back (e.g. `chain.download(&index.out_counts, ...)`)
    /// and verify offsets/counts against the CPU reference in tests.
    pub fn query(
        self: *GridIndex,
        chain: *runtime.Chain,
        queries: []const Point,
        query_count: usize,
        radius: f32,
    ) !void {
        if (query_count == 0 or query_count > self.max_queries or query_count > queries.len) {
            return error.GpuError;
        }

        var params = GpuParams{
            .min_x = self.grid.min.x,
            .min_y = self.grid.min.y,
            .min_z = self.grid.min.z,
            .cell_size = self.grid.cell_size,
            .gx = self.grid.gx,
            .gy = self.grid.gy,
            .gz = self.grid.gz,
            .point_count = 0,
            .query_count = @intCast(query_count),
            .radius = radius,
        };
        self.params_query.markHostDirty();
        try self.params_query.toDevice(self.ctx, std.mem.asBytes(&params));
        self.queries_buf.markHostDirty();
        try self.queries_buf.toDevice(self.ctx, std.mem.sliceAsBytes(queries[0..query_count]));

        const query_grid = try self.query_kernel.gridLinear(query_count);
        try chain.dispatch(&self.query_kernel, self.bind_query.?, &.{
            &self.points_buf,
            &self.counts,
            &self.offsets,
            &self.slots,
            &self.queries_buf,
            &self.out_counts,
            &self.params_query,
        }, query_grid);
    }
};

// ---- tests ----

const TestScene = struct {
    grid: reference.Grid,
    points: []Point,
    xs: []f32,
    ys: []f32,
    zs: []f32,
    queries: []Point,
    qx: []f32,
    qy: []f32,
    qz: []f32,
};

fn makeScene(allocator: std.mem.Allocator, n: usize, q: usize, seed: u64) !TestScene {
    const points = try allocator.alloc(Point, n);
    const xs = try allocator.alloc(f32, n);
    const ys = try allocator.alloc(f32, n);
    const zs = try allocator.alloc(f32, n);
    const queries = try allocator.alloc(Point, q);
    const qx = try allocator.alloc(f32, q);
    const qy = try allocator.alloc(f32, q);
    const qz = try allocator.alloc(f32, q);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (points, 0..) |*point, index| {
        // A few points sit outside the grid on purpose (must be ignored).
        const outside = index % 97 == 0;
        const scale: f32 = if (outside) 14.0 else 10.0;
        const x = random.float(f32) * scale;
        const y = random.float(f32) * scale;
        const z = random.float(f32) * scale;
        point.* = .{ x, y, z, 1.0 };
        xs[index] = x;
        ys[index] = y;
        zs[index] = z;
    }
    for (queries, 0..) |*query, index| {
        const x = 1.0 + random.float(f32) * 8.0;
        const y = 1.0 + random.float(f32) * 8.0;
        const z = 1.0 + random.float(f32) * 8.0;
        query.* = .{ x, y, z, 1.0 };
        qx[index] = x;
        qy[index] = y;
        qz[index] = z;
    }

    return .{
        .grid = .{
            .min = .{ .x = 0, .y = 0, .z = 0 },
            .cell_size = 10.0 / 16.0,
            .gx = 16,
            .gy = 16,
            .gz = 16,
        },
        .points = points,
        .xs = xs,
        .ys = ys,
        .zs = zs,
        .queries = queries,
        .qx = qx,
        .qy = qy,
        .qz = qz,
    };
}

fn freeScene(allocator: std.mem.Allocator, scene: *const TestScene) void {
    allocator.free(scene.points);
    allocator.free(scene.xs);
    allocator.free(scene.ys);
    allocator.free(scene.zs);
    allocator.free(scene.queries);
    allocator.free(scene.qx);
    allocator.free(scene.qy);
    allocator.free(scene.qz);
}

test "gpu grid build and radius query match the cpu reference" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    const n: usize = 4096;
    const q: usize = 256;
    const radius: f32 = 0.9;
    var scene = try makeScene(gpa, n, q, 31);
    defer freeScene(gpa, &scene);

    // CPU reference: build + query (SoA inputs).
    const cells = scene.grid.cellCount();
    const cpu = struct {
        fn run(allocator: std.mem.Allocator, sc: *const TestScene, r: f32) ![]u32 {
            const counts = try allocator.alloc(u32, sc.grid.cellCount());
            defer allocator.free(counts);
            const offsets = try allocator.alloc(u32, sc.grid.cellCount());
            defer allocator.free(offsets);
            const cursor = try allocator.alloc(u32, sc.grid.cellCount());
            defer allocator.free(cursor);
            const slots = try allocator.alloc(u32, sc.points.len);
            defer allocator.free(slots);
            const out = try allocator.alloc(u32, sc.queries.len);
            errdefer allocator.free(out);

            reference.build(sc.grid, sc.xs, sc.ys, sc.zs, counts, offsets, cursor, slots);
            reference.queryCounts(sc.grid, sc.xs, sc.ys, sc.zs, counts, offsets, slots, sc.qx, sc.qy, sc.qz, r, out);
            return out;
        }
    }.run;
    const expected = try cpu(gpa, &scene, radius);
    defer gpa.free(expected);

    // GPU: one chain = build + query, one submit, two readbacks.
    var index = try GridIndex.init(ctx, scene.grid, n, q);
    defer index.deinit();

    const gpu_counts = try gpa.alloc(u32, q);
    defer gpa.free(gpu_counts);
    const gpu_offsets = try gpa.alloc(u32, cells);
    defer gpa.free(gpu_offsets);
    const gpu_cell_counts = try gpa.alloc(u32, cells);
    defer gpa.free(gpu_cell_counts);
    const gpu_slots = try gpa.alloc(u32, n);
    defer gpa.free(gpu_slots);

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    try index.build(&chain, scene.points, n);
    try index.query(&chain, scene.queries, q, radius);
    try chain.download(&index.out_counts, std.mem.sliceAsBytes(gpu_counts));
    try chain.download(&index.offsets, std.mem.sliceAsBytes(gpu_offsets));
    try chain.download(&index.counts, std.mem.sliceAsBytes(gpu_cell_counts));
    try chain.download(&index.slots, std.mem.sliceAsBytes(gpu_slots));
    try chain.submit();

    // Cells: counts/offsets must match the CPU reference (slot order may differ
    // because the GPU scatter uses atomics).
    const cpu_cell_counts = try gpa.alloc(u32, cells);
    defer gpa.free(cpu_cell_counts);
    const cpu_offsets = try gpa.alloc(u32, cells);
    defer gpa.free(cpu_offsets);
    const cpu_cursor = try gpa.alloc(u32, cells);
    defer gpa.free(cpu_cursor);
    const cpu_slots = try gpa.alloc(u32, n);
    defer gpa.free(cpu_slots);
    reference.build(scene.grid, scene.xs, scene.ys, scene.zs, cpu_cell_counts, cpu_offsets, cpu_cursor, cpu_slots);

    try std.testing.expectEqualSlices(u32, cpu_cell_counts, gpu_cell_counts);
    try std.testing.expectEqualSlices(u32, cpu_offsets, gpu_offsets);

    // `slots` groups point indices by cell; the order inside a cell differs
    // (atomic scatter), so compare each cell's slot set after sorting.  Points
    // outside the grid are not indexed and simply absent from both sides.
    var cell: usize = 0;
    while (cell < cells) : (cell += 1) {
        const start = gpu_offsets[cell];
        const end = start + gpu_cell_counts[cell];
        const gpu_cell_slots = gpu_slots[start..end];
        const cpu_cell_slots = cpu_slots[start..end];
        std.mem.sort(u32, gpu_cell_slots, {}, std.sort.asc(u32));
        std.mem.sort(u32, cpu_cell_slots, {}, std.sort.asc(u32));
        try std.testing.expectEqualSlices(u32, cpu_cell_slots, gpu_cell_slots);
    }

    try std.testing.expectEqualSlices(u32, expected, gpu_counts);
}

test "gpu grid query with degenerate inputs" {
    const gpa = std.testing.allocator;
    const ctx = runtime.open() catch |err| switch (err) {
        error.GpuError => return error.SkipZigTest,
    };

    // Single point, single query, radius 0 -> exactly 1 (point equals query).
    const points = [_]Point{.{ 1.0, 2.0, 3.0, 0.0 }};
    const queries = [_]Point{
        .{ 1.0, 2.0, 3.0, 0.0 },
        .{ 9.0, 9.0, 9.0, 0.0 }, // outside grid -> 0
    };
    const grid = reference.Grid{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .cell_size = 1.0,
        .gx = 8,
        .gy = 8,
        .gz = 8,
    };
    var index = try GridIndex.init(ctx, grid, 8, 8);
    defer index.deinit();

    const out = try gpa.alloc(u32, queries.len);
    defer gpa.free(out);

    var chain = try runtime.Chain.begin(ctx);
    defer chain.deinit();
    try index.build(&chain, &points, points.len);
    try index.query(&chain, &queries, queries.len, 0.0);
    try chain.download(&index.out_counts, std.mem.sliceAsBytes(out));
    try chain.submit();

    try std.testing.expectEqual(@as(u32, 1), out[0]);
    try std.testing.expectEqual(@as(u32, 0), out[1]);
}
