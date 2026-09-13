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
) bool {
    switch (backend) {
        .cpu_scalar => {
            computeAccel.ComputeEngine(.cpu_scalar).add(f32, out, a, b);
            return true;
        },
        .cpu_simd => {
            computeAccel.ComputeEngine(.cpu_simd).add(f32, out, a, b);
            return true;
        },
        .gpu_webgpu => {
            computeAccel.gpu.add(out, a, b) catch |err| {
                // A machine without a WebGPU adapter must still produce a
                // correct result rather than turning a manual selection into a
                // panic.  The pipeline records the detailed probe/operation
                // reason; retain the error name as a final safety net.
                if (computeAccel.gpu.lastFallbackReason() == null) {
                    computeAccel.gpu.recordFallback(@errorName(err));
                }
                computeAccel.ComputeEngine(.cpu_simd).add(f32, out, a, b);
                return false;
            };
            return true;
        },
        .gpu_cuda => return false,
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

    a.toDevice();
    b.toDevice();
    var gpu_executed = addWithBackend(chosen, res.cpu_ptr, a.cpu_ptr, b.cpu_ptr);
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

    var gpu_measurement_failed = false;
    var gpu_measurement_reason: ?[]const u8 = null;
    if (chosen == .gpu_webgpu and gpu_executed) {
        const gpu_ns: u64 = computeAccel.bench.timeGpuAdd(
            res.cpu_ptr,
            a.cpu_ptr,
            b.cpu_ptr,
            iters,
        ) catch |err| blk: {
            gpu_measurement_failed = true;
            gpu_measurement_reason =
                computeAccel.gpu.lastFallbackReason() orelse @errorName(err);
            break :blk 0;
        };
        if (!gpu_measurement_failed and gpu_ns != 0) {
            const gpu_speedup = @as(f64, @floatFromInt(simd_ns)) /
                @as(f64, @floatFromInt(gpu_ns));
            try stdout_writer.print(
                "gpu_webgpu  {}   {d:.3}\n",
                .{ gpu_ns, throughputGbps(size, iters, gpu_ns) },
            );
            try stdout_writer.print("speedup (gpu/cpu_simd) = {d:.2}x\n", .{gpu_speedup});
        }

        if (!gpu_measurement_failed and iters > 1) {
            const batch_ns = computeAccel.bench.timeGpuAddBatched(
                res.cpu_ptr,
                a.cpu_ptr,
                b.cpu_ptr,
                iters,
            ) catch |err| blk: {
                const reason = computeAccel.gpu.lastFallbackReason() orelse @errorName(err);
                try stdout_writer.print("gpu_batch measurement failed: {s}\n", .{reason});
                break :blk 0;
            };
            if (batch_ns != 0) {
                const batch_speedup = @as(f64, @floatFromInt(simd_ns)) /
                    @as(f64, @floatFromInt(batch_ns));
                try stdout_writer.print(
                    "gpu_batch    {}   {d:.3}\n",
                    .{ batch_ns, throughputGbps(size, iters, batch_ns) },
                );
                try stdout_writer.print("speedup (gpu_batch/cpu_simd) = {d:.2}x\n", .{batch_speedup});
            }
        }
    }

    // A failed measurement is also a failed GPU execution path for this run.
    // Recompute the visible result with CPU SIMD before reporting the effective
    // backend, so a transient GPU error cannot leave a misleading GPU result.
    if (chosen == .gpu_webgpu and gpu_measurement_failed) {
        gpu_executed = false;
        computeAccel.ComputeEngine(.cpu_simd).add(f32, res.cpu_ptr, a.cpu_ptr, b.cpu_ptr);
        res.toHost();
    }

    if (chosen == .gpu_webgpu) {
        if (gpu_executed) {
            try stdout_writer.print(
                "selected backend = gpu_webgpu  (mode={s}, size={}, iters={})\n",
                .{ modeName(mode), size, iters },
            );
        } else {
            const reason = gpu_measurement_reason orelse
                (computeAccel.gpu.lastFallbackReason() orelse "GPU operation unavailable");
            try stdout_writer.print(
                "selected backend = gpu_webgpu (fell back: {s})  (mode={s}, size={}, iters={})\n",
                .{ reason, modeName(mode), size, iters },
            );
        }
    } else {
        try stdout_writer.print(
            "selected backend = {s}  (mode={s}, size={}, iters={})\n",
            .{ chosen.name(), modeName(mode), size, iters },
        );
    }

    if (mode == .benchmark) {
        if (computeAccel.bench.lastPickReport()) |report| {
            if (!report.gpu_probe.available) {
                try stdout_writer.print(
                    "selection: gpu_webgpu 未纳入端到端 bench（能力探测失败: {s}），保留 {s}\n",
                    .{ report.gpu_probe.reason, report.selected.name() },
                );
            } else if (report.gpu_ns) |measured_gpu_ns| {
                if (report.selected == .gpu_webgpu) {
                    try stdout_writer.print(
                        "selection: gpu_webgpu 探测成功且端到端实测最快（gpu={}ns, cpu_simd={}ns）\n",
                        .{ measured_gpu_ns, report.simd_ns },
                    );
                } else {
                    try stdout_writer.print(
                        "selection: gpu_webgpu 探测成功但端到端实测未胜出（gpu={}ns, cpu_simd={}ns），保留 {s}\n",
                        .{ measured_gpu_ns, report.simd_ns, report.selected.name() },
                    );
                }
            } else {
                try stdout_writer.print(
                    "selection: gpu_webgpu 探测成功但端到端执行失败（{s}），未纳入选择，保留 {s}\n",
                    .{ report.gpu_failure_reason orelse "unknown GPU error", report.selected.name() },
                );
            }
        }
    }

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
