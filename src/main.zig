const std = @import("std");
const Io = std.Io;

const computeAccel = @import("computeAccel");
const BackendType = computeAccel.BackendType;
const SelectionMode = computeAccel.SelectionMode;

const default_size: usize = 1 << 20;
const default_iters: usize = 20;

fn parseBackend(name: []const u8) ?BackendType {
    if (std.mem.eql(u8, name, "cpu_scalar")) return .cpu_scalar;
    if (std.mem.eql(u8, name, "cpu_simd")) return .cpu_simd;
    if (std.mem.eql(u8, name, "gpu_webgpu")) return .gpu_webgpu;
    if (std.mem.eql(u8, name, "gpu_cuda")) return .gpu_cuda;
    return null;
}

fn modeName(mode: SelectionMode) []const u8 {
    return switch (mode) {
        .manual => "manual",
        .heuristic => "heuristic",
        .benchmark => "benchmark",
    };
}

fn addWithBackend(
    backend: BackendType,
    out: []f32,
    a: []const f32,
    b: []const f32,
) void {
    switch (backend) {
        .cpu_scalar => computeAccel.ComputeEngine(.cpu_scalar).add(f32, out, a, b),
        .cpu_simd => computeAccel.ComputeEngine(.cpu_simd).add(f32, out, a, b),
        .gpu_webgpu, .gpu_cuda => @panic("computeAccel: gpu backend not implemented in minimal demo"),
    }
}

fn throughputGbps(size: usize, iters: usize, total_ns: u64) f64 {
    if (total_ns == 0) return 0.0;

    const bytes = @as(f64, @floatFromInt(size)) *
        @as(f64, @floatFromInt(@sizeOf(f32))) *
        2.0 *
        @as(f64, @floatFromInt(iters));
    const seconds = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    return bytes / seconds / 1_000_000_000.0;
}

fn printUsage(writer: *Io.Writer) !void {
    try writer.print(
        "usage: computeAccel [--backend <name> | --auto | --heuristic] [--size <n>] [--iters <n>]\n",
        .{},
    );
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var mode: SelectionMode = .heuristic;
    var manual_backend: BackendType = .cpu_scalar;
    var size: usize = default_size;
    var iters: usize = default_iters;

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--backend")) {
            if (i + 1 >= args.len) {
                try stdout_writer.print("error: --backend requires a backend name\n", .{});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            }
            manual_backend = parseBackend(args[i + 1]) orelse {
                try stdout_writer.print("error: unknown backend '{s}'\n", .{args[i + 1]});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            };
            mode = .manual;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--auto")) {
            mode = .benchmark;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--heuristic")) {
            mode = .heuristic;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--size")) {
            if (i + 1 >= args.len) {
                try stdout_writer.print("error: --size requires a number\n", .{});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            }
            size = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--iters")) {
            if (i + 1 >= args.len) {
                try stdout_writer.print("error: --iters requires a number\n", .{});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            }
            iters = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else {
            try stdout_writer.print("error: unknown argument '{s}'\n", .{arg});
            try printUsage(stdout_writer);
            try stdout_writer.flush();
            return;
        }
    }

    const chosen = try computeAccel.selectBackend(
        arena,
        mode,
        manual_backend,
        f32,
        size,
        iters,
    );

    if (!chosen.isImplemented()) {
        try stdout_writer.print(
            "selected backend = {s}: 该后端未接入，demo 结束\n",
            .{chosen.name()},
        );
        try stdout_writer.flush();
        return;
    }

    var a = try computeAccel.DeviceBuffer(f32).init(arena, chosen, size);
    defer a.deinit();
    var b = try computeAccel.DeviceBuffer(f32).init(arena, chosen, size);
    defer b.deinit();
    var res = try computeAccel.DeviceBuffer(f32).init(arena, chosen, size);
    defer res.deinit();

    @memset(a.cpu_ptr, 2.0);
    @memset(b.cpu_ptr, 3.0);

    try stdout_writer.print(
        "selected backend = {s}  (mode={s}, size={}, iters={})\n",
        .{ chosen.name(), modeName(mode), size, iters },
    );

    a.toDevice();
    b.toDevice();
    addWithBackend(chosen, res.cpu_ptr, a.cpu_ptr, b.cpu_ptr);
    res.toHost();

    const scalar_ns = computeAccel.bench.timeAdd(
        f32,
        .cpu_scalar,
        res.cpu_ptr,
        a.cpu_ptr,
        b.cpu_ptr,
        iters,
    );
    const simd_ns = computeAccel.bench.timeAdd(
        f32,
        .cpu_simd,
        res.cpu_ptr,
        a.cpu_ptr,
        b.cpu_ptr,
        iters,
    );
    const speedup = if (simd_ns == 0)
        0.0
    else
        @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns));

    try stdout_writer.print("\nbackend      total_ns    throughput(GB/s)\n", .{});
    try stdout_writer.print(
        "cpu_scalar   {}   {d:.3}\n",
        .{ scalar_ns, throughputGbps(size, iters, scalar_ns) },
    );
    try stdout_writer.print(
        "cpu_simd     {}   {d:.3}\n",
        .{ simd_ns, throughputGbps(size, iters, simd_ns) },
    );
    try stdout_writer.print("speedup (scalar/simd) = {d:.2}x\n", .{speedup});

    try stdout_writer.print("result sample:", .{});
    const sample_count = @min(size, 4);
    for (0..sample_count) |sample_index| {
        if (sample_index == 0) {
            try stdout_writer.print(" {d:.1}", .{res.cpu_ptr[sample_index]});
        } else {
            try stdout_writer.print(", {d:.1}", .{res.cpu_ptr[sample_index]});
        }
    }
    try stdout_writer.print(" (expected 5.0)\n", .{});

    try stdout_writer.flush();
}
