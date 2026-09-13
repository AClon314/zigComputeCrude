const std = @import("std");
const Io = std.Io;

const computeAccel = @import("computeAccel");
const BackendType = computeAccel.BackendType;
const SelectionMode = computeAccel.SelectionMode;

const default_add_size: usize = 1 << 20;
const default_add_iters: usize = 20;
const default_gemm_dim: usize = 512;
const default_gemm_iters: usize = 5;
const default_reduce_size: usize = 1 << 22;
const default_reduce_iters: usize = 10;

const Kernel = enum { add, gemm, reduce };
const VariantChoice = enum { simple, tiled, both };

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
        };
    }

    switch (kernel) {
        .add => try runAddDemo(stdout_writer, arena, mode, manual_backend, size, iters),
        .gemm => try runGemmDemo(stdout_writer, arena, m, k, n, iters, variant_choice),
        .reduce => try runReduceDemo(stdout_writer, arena, size, iters, reduce_op),
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
