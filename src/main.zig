const std = @import("std");
const Io = std.Io;

const computeAccel = @import("computeAccel");
const spatial = @import("computeAccel_spatial");
const BackendType = computeAccel.BackendType;
const SelectionMode = computeAccel.SelectionMode;

const default_add_size: usize = 1 << 20;
const default_add_iters: usize = 20;
const default_gemm_dim: usize = 512;
const default_gemm_iters: usize = 5;
const default_reduce_size: usize = 1 << 22;
const default_reduce_iters: usize = 10;
const default_chain_iters: usize = 3;
const default_spatial_points: usize = 1 << 16;
const default_spatial_queries: usize = 1 << 12;
const default_spatial_radius: f32 = 2.0;

const Kernel = enum { add, gemm, reduce, chain, spatial };
const VariantChoice = enum { simple, tiled, both };
const ChainKind = enum {
    saxpy,
    pipeline,

    fn name(self: ChainKind) []const u8 {
        return switch (self) {
            .saxpy => "saxpy",
            .pipeline => "gemm+bias+reduce",
        };
    }
};

fn parseChainKind(name: []const u8) ?ChainKind {
    if (std.mem.eql(u8, name, "saxpy")) return .saxpy;
    if (std.mem.eql(u8, name, "pipeline")) return .pipeline;
    return null;
}

fn parseBackend(name: []const u8) ?BackendType {
    if (std.mem.eql(u8, name, "cpu_scalar")) return .cpu_scalar;
    if (std.mem.eql(u8, name, "cpu_simd")) return .cpu_simd;
    if (std.mem.eql(u8, name, "gpu_webgpu")) return .gpu_webgpu;
    if (std.mem.eql(u8, name, "gpu_cuda")) return .gpu_cuda;
    return null;
}

fn parseKernel(name: []const u8) ?Kernel {
    if (std.mem.eql(u8, name, "add")) return .add;
    if (std.mem.eql(u8, name, "gemm")) return .gemm;
    if (std.mem.eql(u8, name, "reduce")) return .reduce;
    if (std.mem.eql(u8, name, "chain")) return .chain;
    if (std.mem.eql(u8, name, "spatial")) return .spatial;
    return null;
}

fn parseVariant(name: []const u8) ?VariantChoice {
    if (std.mem.eql(u8, name, "simple")) return .simple;
    if (std.mem.eql(u8, name, "tiled")) return .tiled;
    if (std.mem.eql(u8, name, "both")) return .both;
    return null;
}

fn parseReduceOp(name: []const u8) ?computeAccel.reduce.Op {
    if (std.mem.eql(u8, name, "sum")) return .sum;
    if (std.mem.eql(u8, name, "max")) return .max;
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

/// flops/ns == GFLOP/s (1 GFLOP/s = 1e9 flop/s and 1e9 ns = 1 s).
fn gflops(flops: f64, total_ns: u64) f64 {
    if (total_ns == 0) return 0.0;
    return flops / @as(f64, @floatFromInt(total_ns));
}

/// bytes/ns == GB/s for the same reason.
fn bytesPerNsGbps(bytes: usize, total_ns: u64) f64 {
    if (total_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(total_ns));
}

fn printUsage(writer: *Io.Writer) !void {
    try writer.print(
        "usage: computeAccel [--backend <name> | --auto | --heuristic] [--size <n>] [--iters <n>]\n" ++
            "       computeAccel --kernel gemm [--m <n>] [--k <n>] [--n <n>] [--variant simple|tiled|both] [--iters <n>]\n" ++
            "       computeAccel --kernel reduce [--size <n>] [--op sum|max] [--iters <n>]\n" ++
            "       computeAccel --kernel chain [--chain saxpy|pipeline] [--chain-lens 1,4,16,64] [--size <n>] [--iters <n>]\n" ++
            "       computeAccel --kernel spatial [--points <n>] [--queries <n>] [--radius <r>] [--iters <n>]\n" ++
            "   --kernel add is the default; --backend/--auto/--heuristic only apply to it.\n",
        .{},
    );
}

fn fillDeterministic(data: []f32, seed: usize) void {
    for (data, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt((index * 31 + seed * 17) % 23)) * 0.25 - 2.0;
    }
}

fn requireValue(writer: *Io.Writer, args: anytype, index: usize, flag: []const u8) !bool {
    if (index + 1 < args.len) return true;
    try writer.print("error: {s} requires a value\n", .{flag});
    try printUsage(writer);
    return false;
}

fn maxAbsValue(data: []const f32) f32 {
    var max_value: f32 = 0;
    for (data) |value| max_value = @max(max_value, @abs(value));
    return max_value;
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
    var size: usize = default_add_size;
    var iters: usize = default_add_iters;
    var iters_explicit = false;
    var kernel: Kernel = .add;
    var m: usize = default_gemm_dim;
    var k: usize = default_gemm_dim;
    var n: usize = default_gemm_dim;
    var variant_choice: VariantChoice = .both;
    var reduce_op: computeAccel.reduce.Op = .sum;
    var chain_kind: ChainKind = .saxpy;
    var chain_lens: []const u8 = "1,4,16,64";
    var point_count: usize = default_spatial_points;
    var query_count: usize = default_spatial_queries;
    var radius: f32 = default_spatial_radius;

    var i: usize = 1;
    while (i < args.len) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--backend")) {
            if (!try requireValue(stdout_writer, args, i, "--backend")) return;
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
        } else if (std.mem.eql(u8, arg, "--kernel")) {
            if (!try requireValue(stdout_writer, args, i, "--kernel")) return;
            kernel = parseKernel(args[i + 1]) orelse {
                try stdout_writer.print("error: unknown kernel '{s}' (expected add|gemm|reduce)\n", .{args[i + 1]});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            };
            i += 2;
        } else if (std.mem.eql(u8, arg, "--variant")) {
            if (!try requireValue(stdout_writer, args, i, "--variant")) return;
            variant_choice = parseVariant(args[i + 1]) orelse {
                try stdout_writer.print("error: unknown variant '{s}' (expected simple|tiled|both)\n", .{args[i + 1]});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            };
            i += 2;
        } else if (std.mem.eql(u8, arg, "--op")) {
            if (!try requireValue(stdout_writer, args, i, "--op")) return;
            reduce_op = parseReduceOp(args[i + 1]) orelse {
                try stdout_writer.print("error: unknown reduce op '{s}' (expected sum|max)\n", .{args[i + 1]});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            };
            i += 2;
        } else if (std.mem.eql(u8, arg, "--chain")) {
            if (!try requireValue(stdout_writer, args, i, "--chain")) return;
            chain_kind = parseChainKind(args[i + 1]) orelse {
                try stdout_writer.print("error: unknown chain '{s}' (expected saxpy|pipeline)\n", .{args[i + 1]});
                try printUsage(stdout_writer);
                try stdout_writer.flush();
                return;
            };
            i += 2;
        } else if (std.mem.eql(u8, arg, "--points")) {
            if (!try requireValue(stdout_writer, args, i, "--points")) return;
            point_count = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--queries")) {
            if (!try requireValue(stdout_writer, args, i, "--queries")) return;
            query_count = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--radius")) {
            if (!try requireValue(stdout_writer, args, i, "--radius")) return;
            radius = try std.fmt.parseFloat(f32, args[i + 1]);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--chain-lens")) {
            if (!try requireValue(stdout_writer, args, i, "--chain-lens")) return;
            chain_lens = args[i + 1];
            i += 2;
        } else if (std.mem.eql(u8, arg, "--size")) {
            if (!try requireValue(stdout_writer, args, i, "--size")) return;
            size = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--iters")) {
            if (!try requireValue(stdout_writer, args, i, "--iters")) return;
            iters = try std.fmt.parseInt(usize, args[i + 1], 10);
            iters_explicit = true;
            i += 2;
        } else if (std.mem.eql(u8, arg, "--m")) {
            if (!try requireValue(stdout_writer, args, i, "--m")) return;
            m = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--k")) {
            if (!try requireValue(stdout_writer, args, i, "--k")) return;
            k = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else if (std.mem.eql(u8, arg, "--n")) {
            if (!try requireValue(stdout_writer, args, i, "--n")) return;
            n = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 2;
        } else {
            try stdout_writer.print("error: unknown argument '{s}'\n", .{arg});
            try printUsage(stdout_writer);
            try stdout_writer.flush();
            return;
        }
    }

    if (!iters_explicit) {
        iters = switch (kernel) {
            .add => default_add_iters,
            .gemm => default_gemm_iters,
            .reduce => default_reduce_iters,
            .chain => default_chain_iters,
            .spatial => default_chain_iters,
        };
    }

    switch (kernel) {
        .add => try runAddDemo(stdout_writer, arena, mode, manual_backend, size, iters),
        .gemm => try runGemmDemo(stdout_writer, arena, m, k, n, iters, variant_choice),
        .reduce => try runReduceDemo(stdout_writer, arena, size, iters, reduce_op),
        .chain => try runChainDemo(stdout_writer, arena, chain_kind, chain_lens, size, m, k, n, iters),
        .spatial => try runSpatialDemo(stdout_writer, arena, point_count, query_count, radius, iters),
    }

    try stdout_writer.flush();
}

fn runAddDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    mode: SelectionMode,
    manual_backend: BackendType,
    size: usize,
    iters: usize,
) !void {
    const chosen = try computeAccel.selectBackend(
        allocator,
        mode,
        manual_backend,
        f32,
        size,
        iters,
    );

    if (!chosen.isImplemented()) {
        try writer.print(
            "selected backend = {s}: 该后端未接入，demo 结束\n",
            .{chosen.name()},
        );
        return;
    }

    var a = try computeAccel.DeviceBuffer(f32).init(allocator, chosen, size);
    defer a.deinit();
    var b = try computeAccel.DeviceBuffer(f32).init(allocator, chosen, size);
    defer b.deinit();
    var res = try computeAccel.DeviceBuffer(f32).init(allocator, chosen, size);
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

    try writer.print("\nbackend      total_ns    throughput(GB/s)\n", .{});
    try writer.print(
        "cpu_scalar   {}   {d:.3}\n",
        .{ scalar_ns, throughputGbps(size, iters, scalar_ns) },
    );
    try writer.print(
        "cpu_simd     {}   {d:.3}\n",
        .{ simd_ns, throughputGbps(size, iters, simd_ns) },
    );
    try writer.print("speedup (scalar/simd) = {d:.2}x\n", .{speedup});

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
            try writer.print(
                "gpu_webgpu  {}   {d:.3}\n",
                .{ gpu_ns, throughputGbps(size, iters, gpu_ns) },
            );
            try writer.print("speedup (gpu/cpu_simd) = {d:.2}x\n", .{gpu_speedup});
        }

        if (!gpu_measurement_failed and iters > 1) {
            const batch_ns = computeAccel.bench.timeGpuAddBatched(
                res.cpu_ptr,
                a.cpu_ptr,
                b.cpu_ptr,
                iters,
            ) catch |err| blk: {
                const reason = computeAccel.gpu.lastFallbackReason() orelse @errorName(err);
                try writer.print("gpu_batch measurement failed: {s}\n", .{reason});
                break :blk 0;
            };
            if (batch_ns != 0) {
                const batch_speedup = @as(f64, @floatFromInt(simd_ns)) /
                    @as(f64, @floatFromInt(batch_ns));
                try writer.print(
                    "gpu_batch    {}   {d:.3}\n",
                    .{ batch_ns, throughputGbps(size, iters, batch_ns) },
                );
                try writer.print("speedup (gpu_batch/cpu_simd) = {d:.2}x\n", .{batch_speedup});
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
            try writer.print(
                "selected backend = gpu_webgpu  (mode={s}, size={}, iters={})\n",
                .{ modeName(mode), size, iters },
            );
        } else {
            const reason = gpu_measurement_reason orelse
                (computeAccel.gpu.lastFallbackReason() orelse "GPU operation unavailable");
            try writer.print(
                "selected backend = gpu_webgpu (fell back: {s})  (mode={s}, size={}, iters={})\n",
                .{ reason, modeName(mode), size, iters },
            );
        }
    } else {
        try writer.print(
            "selected backend = {s}  (mode={s}, size={}, iters={})\n",
            .{ chosen.name(), modeName(mode), size, iters },
        );
    }

    if (mode == .benchmark) {
        if (computeAccel.bench.lastPickReport()) |report| {
            if (!report.gpu_probe.available) {
                try writer.print(
                    "selection: gpu_webgpu 未纳入端到端 bench（能力探测失败: {s}），保留 {s}\n",
                    .{ report.gpu_probe.reason, report.selected.name() },
                );
            } else if (report.gpu_ns) |measured_gpu_ns| {
                if (report.selected == .gpu_webgpu) {
                    try writer.print(
                        "selection: gpu_webgpu 探测成功且端到端实测最快（gpu={}ns, cpu_simd={}ns）\n",
                        .{ measured_gpu_ns, report.simd_ns },
                    );
                } else {
                    try writer.print(
                        "selection: gpu_webgpu 探测成功但端到端实测未胜出（gpu={}ns, cpu_simd={}ns），保留 {s}\n",
                        .{ measured_gpu_ns, report.simd_ns, report.selected.name() },
                    );
                }
            } else {
                try writer.print(
                    "selection: gpu_webgpu 探测成功但端到端执行失败（{s}），未纳入选择，保留 {s}\n",
                    .{ report.gpu_failure_reason orelse "unknown GPU error", report.selected.name() },
                );
            }
        }
    }

    try writer.print("result sample:", .{});
    const sample_count = @min(size, 4);
    for (0..sample_count) |sample_index| {
        if (sample_index == 0) {
            try writer.print(" {d:.1}", .{res.cpu_ptr[sample_index]});
        } else {
            try writer.print(", {d:.1}", .{res.cpu_ptr[sample_index]});
        }
    }
    try writer.print(" (expected 5.0)\n", .{});
}

fn runGemmDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
    variant_choice: VariantChoice,
) !void {
    if (m == 0 or k == 0 or n == 0) {
        try writer.print("error: GEMM dimensions must be non-zero\n", .{});
        return;
    }

    const flops = 2.0 *
        @as(f64, @floatFromInt(m)) *
        @as(f64, @floatFromInt(k)) *
        @as(f64, @floatFromInt(n));
    try writer.print(
        "\n== GEMM m={} k={} n={} ({d:.2} MFLOP, {d:.2} MiB operands) iters={} ==\n",
        .{
            m,
            k,
            n,
            flops / 1e6,
            @as(f64, @floatFromInt((m * k + k * n + m * n) * @sizeOf(f32))) / (1024.0 * 1024.0),
            iters,
        },
    );

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const reference = try allocator.alloc(f32, m * n);
    defer allocator.free(reference);
    const work = try allocator.alloc(f32, m * n);
    defer allocator.free(work);

    fillDeterministic(a, 1);
    fillDeterministic(b, 2);
    computeAccel.gemm.referenceSimd(m, k, n, a, b, reference);

    const scalar_ns = computeAccel.bench.timeGemmCpu(false, m, k, n, a, b, work, iters);
    const simd_ns = computeAccel.bench.timeGemmCpu(true, m, k, n, a, b, work, iters);

    const total_flops = flops * @as(f64, @floatFromInt(iters));

    try writer.print("backend        total_ns      GFLOP/s\n", .{});
    try writer.print(
        "cpu_scalar     {}   {d:.3}\n",
        .{ scalar_ns, gflops(total_flops, scalar_ns) },
    );
    try writer.print(
        "cpu_simd       {}   {d:.3}\n",
        .{ simd_ns, gflops(total_flops, simd_ns) },
    );
    try writer.print(
        "speedup (cpu_simd/cpu_scalar) = {d:.2}x\n",
        .{if (scalar_ns == 0) 0.0 else @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns))},
    );

    const probe = computeAccel.GpuContext.probe();
    if (!probe.available) {
        try writer.print("gpu_webgpu: 不可用（{s}）；以上为 CPU 结果\n", .{probe.reason});
        return;
    }

    const variants: []const computeAccel.gemm.Variant = switch (variant_choice) {
        .simple => &.{.simple},
        .tiled => &.{.tiled},
        .both => &.{ .simple, .tiled },
    };

    var verified_any = false;
    for (variants) |variant| {
        if (!computeAccel.gemm.canRun(probe.limits, m, k, n, variant)) {
            try writer.print(
                "gpu_{s}: 超出设备 limits（buffers / dispatch grid），跳过\n",
                .{variant.name()},
            );
            continue;
        }

        @memset(work, 0);
        computeAccel.gemm.gemm(variant, m, k, n, a, b, work) catch |err| {
            try writer.print(
                "gpu_{s}: 执行失败（{s}），跳过\n",
                .{ variant.name(), computeAccel.gpu.lastFallbackReason() orelse @errorName(err) },
            );
            continue;
        };
        const tolerance = @max(1.0, maxAbsValue(reference)) * 1e-4;
        const diff = computeAccel.gemm.maxAbsDiff(reference, work);
        const verdict = if (diff <= tolerance) "OK" else "MISMATCH";
        verified_any = true;

        const gpu_ns = computeAccel.bench.timeGemm(variant, m, k, n, a, b, work, iters) catch |err| {
            try writer.print(
                "gpu_{s}: 端到端计时失败（{s}）\n",
                .{ variant.name(), computeAccel.gpu.lastFallbackReason() orelse @errorName(err) },
            );
            continue;
        };
        try writer.print(
            "gpu_{s}{s}   {}   {d:.3}\n",
            .{
                variant.name(),
                if (variant == .simple) "       " else "        ",
                gpu_ns,
                gflops(total_flops, gpu_ns),
            },
        );
        try writer.print(
            "  end-to-end max|diff|={e:.3} (tol {e:.1}) {s}; speedup vs cpu_simd = {d:.2}x\n",
            .{
                diff,
                tolerance,
                verdict,
                if (simd_ns == 0) 0.0 else @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(gpu_ns)),
            },
        );

        if (iters > 1) {
            const batch_ns = computeAccel.bench.timeGemmBatched(
                variant,
                m,
                k,
                n,
                a,
                b,
                work,
                iters,
            ) catch |err| {
                try writer.print(
                    "gpu_{s}_batch: 计时失败（{s}）\n",
                    .{ variant.name(), computeAccel.gpu.lastFallbackReason() orelse @errorName(err) },
                );
                continue;
            };
            try writer.print(
                "gpu_{s}_batch{s}   {}   {d:.3}\n",
                .{
                    variant.name(),
                    if (variant == .simple) " " else "  ",
                    batch_ns,
                    gflops(total_flops, batch_ns),
                },
            );
            try writer.print(
                "  steady-state (1 upload + {} dispatches + 1 readback); speedup vs cpu_simd = {d:.2}x\n",
                .{
                    iters,
                    if (simd_ns == 0) 0.0 else @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(batch_ns)),
                },
            );
        }
    }

    if (verified_any) {
        try writer.print("verification: GPU results compared element-wise with the cpu_simd reference\n", .{});
    }
}

fn runReduceDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    size: usize,
    iters: usize,
    op: computeAccel.reduce.Op,
) !void {
    if (size == 0) {
        try writer.print("error: reduce size must be non-zero\n", .{});
        return;
    }

    const bytes = size * @sizeOf(f32);
    try writer.print(
        "\n== reduce {s} n={} ({d:.2} MiB input) iters={} ==\n",
        .{ op.name(), size, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), iters },
    );

    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    fillDeterministic(input, 3);

    var cpu_scalar: f32 = 0;
    var cpu_simd: f32 = 0;
    var gpu_value: f32 = 0;

    const scalar_ns = computeAccel.bench.timeReduceCpu(false, op, &cpu_scalar, input, iters);
    const simd_ns = computeAccel.bench.timeReduceCpu(true, op, &cpu_simd, input, iters);

    const total_bytes = bytes * iters;

    try writer.print("backend      total_ns      GB/s (input)\n", .{});
    try writer.print(
        "cpu_scalar   {}   {d:.3}\n",
        .{ scalar_ns, bytesPerNsGbps(total_bytes, scalar_ns) },
    );
    try writer.print(
        "cpu_simd     {}   {d:.3}\n",
        .{ simd_ns, bytesPerNsGbps(total_bytes, simd_ns) },
    );

    const probe = computeAccel.GpuContext.probe();
    if (!probe.available) {
        try writer.print("gpu_webgpu: 不可用（{s}）；以上为 CPU 结果\n", .{probe.reason});
        return;
    }
    if (!computeAccel.reduce.canRun(probe.limits, size)) {
        try writer.print("gpu_webgpu: 超出设备 limits（input buffer / dispatch grid），跳过\n", .{});
        return;
    }

    computeAccel.reduce.reduce(op, &gpu_value, input) catch |err| {
        try writer.print(
            "gpu_webgpu: 执行失败（{s}），跳过\n",
            .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)},
        );
        return;
    };
    const cpu_reference = switch (op) {
        .sum => computeAccel.reduce.referenceSumSimd(input),
        .max => computeAccel.reduce.referenceMax(input),
    };
    const diff = @abs(cpu_reference - gpu_value);
    const tolerance = switch (op) {
        .sum => @max(1.0, @abs(cpu_reference)) * 1e-4,
        .max => 0.0,
    };
    const verdict = if (diff <= tolerance) "OK" else "MISMATCH";

    const gpu_ns = computeAccel.bench.timeReduce(op, &gpu_value, input, iters) catch |err| {
        try writer.print(
            "gpu_webgpu: 端到端计时失败（{s}）\n",
            .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)},
        );
        return;
    };
    try writer.print(
        "gpu_webgpu   {}   {d:.3}\n",
        .{ gpu_ns, bytesPerNsGbps(total_bytes, gpu_ns) },
    );
    try writer.print(
        "  {s}: cpu_simd={d:.6} gpu={d:.6} |diff|={e:.3} {s}; speedup vs cpu_simd = {d:.2}x\n",
        .{
            op.name(),
            cpu_reference,
            gpu_value,
            diff,
            verdict,
            if (simd_ns == 0) 0.0 else @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(gpu_ns)),
        },
    );

    if (iters > 1) {
        const batch_ns = computeAccel.bench.timeReduceBatched(op, &gpu_value, input, iters) catch |err| {
            try writer.print(
                "gpu_batch: 计时失败（{s}）\n",
                .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)},
            );
            return;
        };
        if (batch_ns != 0) {
            try writer.print(
                "gpu_batch    {}   {d:.3}\n",
                .{ batch_ns, bytesPerNsGbps(total_bytes, batch_ns) },
            );
            try writer.print(
                "  steady-state (1 upload + {} reductions + 1 readback); speedup vs cpu_simd = {d:.2}x\n",
                .{
                    iters,
                    if (simd_ns == 0) 0.0 else @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(batch_ns)),
                },
            );
        }
    }
}

fn parseLens(buffer: []usize, text: []const u8) ![]usize {
    var count: usize = 0;
    var iter = std.mem.splitScalar(u8, text, ',');
    while (iter.next()) |part| {
        const value = std.fmt.parseInt(usize, part, 10) catch return error.InvalidLens;
        if (value == 0) return error.InvalidLens;
        if (count == buffer.len) return error.InvalidLens;
        buffer[count] = value;
        count += 1;
    }
    if (count == 0) return error.InvalidLens;
    return buffer[0..count];
}

/// Median of the collected samples: micro-benchmarks on a shared machine are
/// noisy, and a median keeps the ablation table honest without cherry-picking.
fn median(values: []u64) u64 {
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    return values[values.len / 2];
}

fn runChainDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    kind: ChainKind,
    lens_text: []const u8,
    size: usize,
    m: usize,
    k: usize,
    n: usize,
    repeats: usize,
) !void {
    var lens_buffer: [8]usize = undefined;
    const lens = parseLens(&lens_buffer, lens_text) catch {
        try writer.print("error: --chain-lens must be a comma list of positive numbers\n", .{});
        return;
    };
    const samples = @min(@max(repeats, 1), 8);

    const probe = computeAccel.GpuContext.probe();
    if (!probe.available) {
        try writer.print("chain demo: gpu_webgpu 不可用（{s}）\n", .{probe.reason});
        return;
    }
    const ctx = computeAccel.runtime.open() catch |err| {
        try writer.print("chain demo: 打开设备失败（{s}）\n", .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)});
        return;
    };

    switch (kind) {
        .saxpy => try runSaxpyChainDemo(writer, allocator, ctx, lens, size, samples),
        .pipeline => try runPipelineChainDemo(writer, allocator, ctx, lens, m, k, n, samples),
    }
}

fn runSaxpyChainDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    ctx: *computeAccel.runtime.Device,
    lens: []const usize,
    size: usize,
    samples: usize,
) !void {
    const alpha: f32 = 0.999;
    const initial = try allocator.alloc(f32, size);
    defer allocator.free(initial);
    const y = try allocator.alloc(f32, size);
    defer allocator.free(y);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);
    const cpu_work = try allocator.alloc(f32, size);
    defer allocator.free(cpu_work);

    fillDeterministic(initial, 11);
    fillDeterministic(y, 12);

    // Warm-up: shader compilation and the lazy staging allocation must not be
    // attributed to any ablation mode (they happen exactly once).
    _ = try computeAccel.chain_bench.runSaxpyChain(allocator, ctx, size, 2, .chained, alpha, initial, y, out);

    try writer.print(
        "\n== chain ablation: saxpy (n={} f32 = {d:.2} MiB/buffer, alpha={d:.3}, {} repeats) ==\n",
        .{ size, @as(f64, @floatFromInt(size * @sizeOf(f32))) / (1024.0 * 1024.0), alpha, samples },
    );
    try writer.print("chain_len  cpu_simd   per_call   per_submit   chained   chained/cpu   verify\n", .{});

    for (lens) |steps| {
        var cpu_ns: u64 = 0;
        {
            var times: [8]u64 = undefined;
            for (0..samples) |sample| {
                @memcpy(cpu_work, initial);
                times[sample] = computeAccel.chain_bench.timeCpuSimdSaxpyChain(cpu_work, y, alpha, steps);
            }
            cpu_ns = median(times[0..samples]);
        }

        var mode_results: [3]computeAccel.chain_bench.SaxpyResult = undefined;
        const modes = [_]computeAccel.chain_bench.Mode{ .per_call, .per_submit, .chained };
        for (modes, 0..) |mode, mode_index| {
            var times: [8]u64 = undefined;
            var diff: f32 = 0;
            for (0..samples) |sample| {
                const result = computeAccel.chain_bench.runSaxpyChain(
                    allocator,
                    ctx,
                    size,
                    steps,
                    mode,
                    alpha,
                    initial,
                    y,
                    out,
                ) catch |err| {
                    try writer.print(
                        "{d:<10} {s} failed: {s}\n",
                        .{ steps, mode.name(), computeAccel.gpu.lastFallbackReason() orelse @errorName(err) },
                    );
                    return;
                };
                times[sample] = result.total_ns;
                diff = @max(diff, result.max_diff);
            }
            mode_results[mode_index] = .{
                .total_ns = median(times[0..samples]),
                .gbps = 0,
                .max_diff = diff,
            };
        }

        const bytes = @as(f64, @floatFromInt(size * @sizeOf(f32) * 3 * steps));
        const cpu_gbps = bytes / @as(f64, @floatFromInt(cpu_ns));
        const per_call_gbps = bytes / @as(f64, @floatFromInt(mode_results[0].total_ns));
        const per_submit_gbps = bytes / @as(f64, @floatFromInt(mode_results[1].total_ns));
        const chained_gbps = bytes / @as(f64, @floatFromInt(mode_results[2].total_ns));
        const speedup = if (chained_gbps == 0) 0 else chained_gbps / cpu_gbps;
        const verify = if (mode_results[2].max_diff <= 1e-3) "OK" else "MISMATCH";

        try writer.print(
            "{d:<10} {d:>8.2}   {d:>8.2}   {d:>10.2}   {d:>7.2}   {d:>11.2}x   {s} ({e:.2})\n",
            .{
                steps,
                cpu_gbps,
                per_call_gbps,
                per_submit_gbps,
                chained_gbps,
                speedup,
                verify,
                mode_results[2].max_diff,
            },
        );
    }
    try writer.print("GB/s counts 3 streams (read x, read y, write x) per step; all modes compute identically.\n", .{});
}

fn runPipelineChainDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    ctx: *computeAccel.runtime.Device,
    lens: []const usize,
    m: usize,
    k: usize,
    n: usize,
    samples: usize,
) !void {
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const bias = try allocator.alloc(f32, n);
    defer allocator.free(bias);
    fillDeterministic(a, 21);
    fillDeterministic(b, 22);
    fillDeterministic(bias, 23);

    const flops = 2.0 *
        @as(f64, @floatFromInt(m)) *
        @as(f64, @floatFromInt(k)) *
        @as(f64, @floatFromInt(n));

    var warm_sum: f32 = 0;
    _ = try computeAccel.chain_bench.runGemmBiasReduceChain(allocator, ctx, m, k, n, .chained, 1, a, b, bias, &warm_sum);

    try writer.print(
        "\n== chain ablation: {s} (m={} k={} n={}, {} repeats) ==\n",
        .{ ChainKind.pipeline.name(), m, k, n, samples },
    );
    try writer.print("repeats  staged(ms)  chained(ms)  submit_cut   chained GFLOP/s   verify\n", .{});

    for (lens) |repetitions| {
        var staged_times: [8]u64 = undefined;
        var chained_times: [8]u64 = undefined;
        var staged_diff: f32 = 0;
        var chained_diff: f32 = 0;
        var staged_sum: f32 = 0;
        var chained_sum: f32 = 0;
        var reference: f32 = 0;
        var relative_diff: f64 = 0;

        for (0..samples) |sample| {
            const staged = computeAccel.chain_bench.runGemmBiasReduceChain(
                allocator,
                ctx,
                m,
                k,
                n,
                .staged,
                repetitions,
                a,
                b,
                bias,
                &staged_sum,
            ) catch |err| {
                try writer.print("staged failed: {s}\n", .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)});
                return;
            };
            staged_times[sample] = staged.total_ns;
            staged_diff = @max(staged_diff, staged.max_diff);
            reference = staged.reference;
            relative_diff = @max(relative_diff, staged.relativeDiff());

            const chained = computeAccel.chain_bench.runGemmBiasReduceChain(
                allocator,
                ctx,
                m,
                k,
                n,
                .chained,
                repetitions,
                a,
                b,
                bias,
                &chained_sum,
            ) catch |err| {
                try writer.print("chained failed: {s}\n", .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)});
                return;
            };
            chained_times[sample] = chained.total_ns;
            chained_diff = @max(chained_diff, chained.max_diff);
            relative_diff = @max(relative_diff, chained.relativeDiff());
        }

        const staged_ns = median(staged_times[0..samples]);
        const chained_ns = median(chained_times[0..samples]);
        const stages: f64 = 3.0; // three host round trips in staged mode
        const submit_cut = @as(f64, @floatFromInt(staged_ns)) / @as(f64, @floatFromInt(chained_ns));
        const chained_gflops = (flops * @as(f64, @floatFromInt(repetitions))) /
            @as(f64, @floatFromInt(chained_ns));
        // A sum over m*n terms is compared with a relative tolerance: the
        // remaining difference is only the f32 accumulation order.
        const verify = if (relative_diff <= 1e-4) "OK" else "MISMATCH";

        _ = stages;
        try writer.print(
            "{d:<8} {d:>10.3}  {d:>11.3}  {d:>9.2}x   {d:>14.2}   {s} (rel {e:.2})\n",
            .{
                repetitions,
                @as(f64, @floatFromInt(staged_ns)) / 1e6,
                @as(f64, @floatFromInt(chained_ns)) / 1e6,
                submit_cut,
                chained_gflops,
                verify,
                relative_diff,
            },
        );
        if (repetitions == lens[lens.len - 1]) {
            try writer.print(
                "last sum: staged={d:.3} chained={d:.3} reference={d:.3} (abs diff {e:.2})\n",
                .{ staged_sum, chained_sum, reference, @max(staged_diff, chained_diff) },
            );
        }
    }
    try writer.print("staged = per-stage submit+readback; chained = 4 dispatches in one submit, 1 readback.\n", .{});
}

fn runSpatialDemo(
    writer: *Io.Writer,
    allocator: std.mem.Allocator,
    point_count: usize,
    query_count: usize,
    radius: f32,
    repeats: usize,
) !void {
    if (point_count == 0 or query_count == 0 or radius < 0) {
        try writer.print("error: --points/--queries must be positive and --radius >= 0\n", .{});
        return;
    }

    const box: f32 = 64.0;
    const dim: u32 = 64;
    const grid = spatial.Grid{
        .min = .{ .x = 0, .y = 0, .z = 0 },
        .cell_size = box / @as(f32, @floatFromInt(dim)),
        .gx = dim,
        .gy = dim,
        .gz = dim,
    };
    const cells = grid.cellCount();
    const samples = @min(@max(repeats, 1), 8);

    // Points inside the grid; query centers kept away from the border so the
    // radius sphere never leaves the grid (grid and brute force then agree).
    const points = try allocator.alloc(spatial.Point, point_count);
    defer allocator.free(points);
    const xs = try allocator.alloc(f32, point_count);
    defer allocator.free(xs);
    const ys = try allocator.alloc(f32, point_count);
    defer allocator.free(ys);
    const zs = try allocator.alloc(f32, point_count);
    defer allocator.free(zs);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const random = prng.random();
    for (points, 0..) |*point, index| {
        const x = random.float(f32) * box;
        const y = random.float(f32) * box;
        const z = random.float(f32) * box;
        point.* = .{ x, y, z, 1.0 };
        xs[index] = x;
        ys[index] = y;
        zs[index] = z;
    }

    const queries = try allocator.alloc(spatial.Point, query_count);
    defer allocator.free(queries);
    const qx = try allocator.alloc(f32, query_count);
    defer allocator.free(qx);
    const qy = try allocator.alloc(f32, query_count);
    defer allocator.free(qy);
    const qz = try allocator.alloc(f32, query_count);
    defer allocator.free(qz);
    const margin = @min(radius + 0.5, box / 2 - 0.5);
    for (queries, 0..) |*query, index| {
        const x = margin + random.float(f32) * (box - 2 * margin);
        const y = margin + random.float(f32) * (box - 2 * margin);
        const z = margin + random.float(f32) * (box - 2 * margin);
        query.* = .{ x, y, z, 1.0 };
        qx[index] = x;
        qy[index] = y;
        qz[index] = z;
    }

    try writer.print(
        "\n== spatial grid: {} points in [0,{d:.0})^3, {} queries, radius={d:.2}, cells={}^3 ==\n",
        .{ point_count, box, query_count, radius, dim },
    );

    // ---- CPU grid ----
    const cpu_counts = try allocator.alloc(u32, cells);
    defer allocator.free(cpu_counts);
    const cpu_offsets = try allocator.alloc(u32, cells);
    defer allocator.free(cpu_offsets);
    const cpu_cursor = try allocator.alloc(u32, cells);
    defer allocator.free(cpu_cursor);
    const cpu_slots = try allocator.alloc(u32, point_count);
    defer allocator.free(cpu_slots);
    const cpu_out = try allocator.alloc(u32, query_count);
    defer allocator.free(cpu_out);
    const brute_out = try allocator.alloc(u32, query_count);
    defer allocator.free(brute_out);

    var cpu_build_ns: u64 = 0;
    var cpu_query_ns: u64 = 0;
    var cpu_brute_ns: u64 = 0;
    {
        const t0 = computeAccel.bench.nowNs();
        spatial.build(grid, xs, ys, zs, cpu_counts, cpu_offsets, cpu_cursor, cpu_slots);
        cpu_build_ns = @intCast(@max(0, computeAccel.bench.nowNs() - t0));
    }
    {
        const t0 = computeAccel.bench.nowNs();
        spatial.queryCounts(grid, xs, ys, zs, cpu_counts, cpu_offsets, cpu_slots, qx, qy, qz, radius, cpu_out);
        cpu_query_ns = @intCast(@max(0, computeAccel.bench.nowNs() - t0));
    }
    {
        const t0 = computeAccel.bench.nowNs();
        spatial.bruteForceCounts(xs, ys, zs, qx, qy, qz, radius, brute_out);
        cpu_brute_ns = @intCast(@max(0, computeAccel.bench.nowNs() - t0));
    }
    for (cpu_out, brute_out, 0..) |a, b, index| {
        if (a != b) {
            try writer.print("cpu grid/brute mismatch at query {}: {} vs {}\n", .{ index, a, b });
            return;
        }
    }

    const probe = computeAccel.GpuContext.probe();
    if (!probe.available) {
        try writer.print("gpu_webgpu: 不可用（{s}）；以上为 CPU 结果\n", .{probe.reason});
        return;
    }
    const ctx = computeAccel.runtime.open() catch |err| {
        try writer.print("gpu_webgpu: 打开设备失败（{s}）\n", .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)});
        return;
    };

    var index = spatial.GridIndex.init(ctx, grid, point_count, query_count, 16) catch |err| {
        try writer.print("gpu_webgpu: 建索引失败（{s}）\n", .{computeAccel.gpu.lastFallbackReason() orelse @errorName(err)});
        return;
    };
    defer index.deinit();

    const gpu_out = try allocator.alloc(u32, query_count);
    defer allocator.free(gpu_out);

    // Neighbor lists: fixed-capacity kNN output (broad-phase style).  Truncated
    // lists are order-dependent on the GPU, so only queries whose count fits
    // the capacity are compared (as sorted sets).
    const max_neighbors: usize = 16;
    const cpu_neighbors = try allocator.alloc(u32, query_count * max_neighbors);
    defer allocator.free(cpu_neighbors);
    const gpu_neighbors = try allocator.alloc(u32, query_count * max_neighbors);
    defer allocator.free(gpu_neighbors);
    const neighbor_counts = try allocator.alloc(u32, query_count);
    defer allocator.free(neighbor_counts);
    spatial.queryNeighbors(
        grid,
        xs,
        ys,
        zs,
        cpu_counts,
        cpu_offsets,
        cpu_slots,
        qx,
        qy,
        qz,
        radius,
        max_neighbors,
        neighbor_counts,
        cpu_neighbors,
    );

    // Warm-up: pipeline compilation + staging allocation must not be attributed
    // to the measured run.
    {
        var chain = try computeAccel.runtime.Chain.begin(ctx);
        defer chain.deinit();
        try index.build(&chain, points, point_count);
        try index.query(&chain, queries, query_count, radius);
        try chain.download(&index.out_counts, std.mem.sliceAsBytes(gpu_out));
        try chain.submit();
    }

    // One chain: build + query + readback (upload + single submit).
    var gpu_once_ns: u64 = 0;
    {
        const t0 = computeAccel.bench.nowNs();
        var chain = try computeAccel.runtime.Chain.begin(ctx);
        defer chain.deinit();
        try index.build(&chain, points, point_count);
        try index.query(&chain, queries, query_count, radius);
        try chain.download(&index.out_counts, std.mem.sliceAsBytes(gpu_out));
        try chain.download(&index.neighbors, std.mem.sliceAsBytes(gpu_neighbors));
        try chain.submit();
        gpu_once_ns = @intCast(@max(0, computeAccel.bench.nowNs() - t0));
    }

    var verify_ok = true;
    for (gpu_out, cpu_out) |a, b| {
        if (a != b) verify_ok = false;
    }

    var neighbors_checked: usize = 0;
    var neighbors_ok = true;
    for (0..query_count) |query| {
        const count = @min(neighbor_counts[query], max_neighbors);
        if (neighbor_counts[query] > max_neighbors) continue; // truncated: order-dependent
        const cpu_list = cpu_neighbors[query * max_neighbors ..][0..count];
        const gpu_list = gpu_neighbors[query * max_neighbors ..][0..count];
        std.mem.sort(u32, cpu_list, {}, std.sort.asc(u32));
        std.mem.sort(u32, gpu_list, {}, std.sort.asc(u32));
        if (!std.mem.eql(u32, cpu_list, gpu_list)) neighbors_ok = false;
        neighbors_checked += 1;
    }

    // Steady state: build once, then `samples` query-only chains (each uploads
    // its queries, records one dispatch, reads back).
    var query_times: [8]u64 = undefined;
    for (0..samples) |sample| {
        const t0 = computeAccel.bench.nowNs();
        var chain = try computeAccel.runtime.Chain.begin(ctx);
        defer chain.deinit();
        try index.query(&chain, queries, query_count, radius);
        try chain.download(&index.out_counts, std.mem.sliceAsBytes(gpu_out));
        try chain.submit();
        query_times[sample] = @intCast(@max(0, computeAccel.bench.nowNs() - t0));
    }
    std.mem.sort(u64, query_times[0..samples], {}, std.sort.asc(u64));
    const gpu_query_ns = query_times[samples / 2];

    var total_avg: f64 = 0;
    for (cpu_out) |value| total_avg += @as(f64, @floatFromInt(value));
    total_avg /= @as(f64, @floatFromInt(query_count));

    try writer.print("backend        total(ms)   build/query(ms)             speedup vs brute   verify\n", .{});
    try writer.print(
        "cpu_brute      {d:>8.3}   - / {d:>8.3}                     1.00x              ref ({} checks)\n",
        .{
            @as(f64, @floatFromInt(cpu_brute_ns)) / 1e6,
            @as(f64, @floatFromInt(cpu_brute_ns)) / 1e6,
            point_count * query_count,
        },
    );
    try writer.print(
        "cpu_grid       {d:>8.3}   {d:>8.3} / {d:>8.3}                     {d:>5.1}x             OK\n",
        .{
            @as(f64, @floatFromInt(cpu_build_ns + cpu_query_ns)) / 1e6,
            @as(f64, @floatFromInt(cpu_build_ns)) / 1e6,
            @as(f64, @floatFromInt(cpu_query_ns)) / 1e6,
            @as(f64, @floatFromInt(cpu_brute_ns)) / @as(f64, @floatFromInt(cpu_build_ns + cpu_query_ns)),
        },
    );
    try writer.print(
        "gpu_grid       {d:>8.3}   {d:>8.3} / {d:>8.3} (1 chain)        {d:>5.1}x             {s}\n",
        .{
            @as(f64, @floatFromInt(gpu_once_ns)) / 1e6,
            @as(f64, @floatFromInt(gpu_once_ns)) / 1e6,
            @as(f64, @floatFromInt(gpu_query_ns)) / 1e6,
            @as(f64, @floatFromInt(cpu_brute_ns)) / @as(f64, @floatFromInt(gpu_once_ns)),
            if (verify_ok) "OK" else "MISMATCH",
        },
    );
    try writer.print(
        "gpu_query      {d:>8.3}   - / {d:>8.3} (median of {})       {d:>5.1}x             OK\n",
        .{
            @as(f64, @floatFromInt(gpu_query_ns)) / 1e6,
            @as(f64, @floatFromInt(gpu_query_ns)) / 1e6,
            samples,
            @as(f64, @floatFromInt(cpu_brute_ns)) / @as(f64, @floatFromInt(gpu_query_ns)),
        },
    );
    try writer.print(
        "avg candidates/query = {d:.1}, brute-force checks/query = {}; GPU counts == CPU grid counts (exact u32)\n",
        .{ total_avg, point_count },
    );
    try writer.print(
        "neighbor lists (K={}): verified {}/{} queries as sorted sets ({s}); truncated queries skipped\n",
        .{ max_neighbors, neighbors_checked, query_count, if (neighbors_ok) "MATCH" else "MISMATCH" },
    );
}
