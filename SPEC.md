# computeAccel — 设计契约（Spec）

> Zig **0.16.0**。目标：一个最小 demo，演示「智能 / 手动选择计算后端以加速计算」。
> 本文件是**唯一事实来源**（API 契约），所有 sub-task（luna）必须严格按此实现。

## 目标

- 提供 `BackendType` 枚举：`cpu_scalar` / `cpu_simd` / `gpu_webgpu` / `gpu_cuda`。
  - 本 demo 只**真正实现**两个 CPU 后端（scalar 用标量循环、simd 用 `@Vector`）。
  - 两个 GPU 后端只作为「占位枚举」存在（体现可扩展性），一旦被调用应 `@panic` 提示未实现。
- 提供 `DeviceBuffer(T)`：统一数据容器，负责分配 / 释放，预留 `to_device` / `to_host` 同步接口（CPU 后端为 no-op，GPU 后端为 stub）。
- 提供 `ComputeEngine(comptime bt: BackendType) type`：**基于 comptime 的静态派发**。
  - `add`：逐元素 `out = a + b`。
  - `saxpy`：`out = alpha * x + y`。
  - 用 comptime `switch(bt)`，编译期剥离无关分支，零运行时开销。
- 提供「选择逻辑」：
  - `manual`：用户显式指定。
  - `heuristic`：按工作负载规模启发式（N 大则用 SIMD）。
  - `benchmark`：运行时对每个可用后端实测（bench），选最快的。
- 提供 `test` 块：正确性 + **性能对比**（当前端 scalar vs simd 的耗时 / 加速比）。

## ⚠️ 关键 API 事实（Zig 0.16.0 已验证）

1. **计时**：`std.time.nanoTimestamp` / `std.time.Timer` 已删除。改用：
   ```zig
   var ts: std.posix.timespec = undefined;
   _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
   return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
   ```
   `ts.sec` / `ts.nsec` 类型为 `isize`。
2. **SIMD**：`const V = @Vector(8, T);` 加/乘均可；写回用：
   ```zig
   const vr: V = va + vb;
   @memcpy(out[i..][0..8], &@as([8]T, vr));
   ```
   `@splat(alpha)` 只用于标量广播：`const va: V = @splat(alpha);`。
3. **局部变量**：不变则用 `const`，Zig 0.16 对「从不 mutated 的 var」报错。
4. `@memset`、`@memcpy`、`0..len` 范围循环均为标准 API。

---

## 模块划分与职责

```
src/
  root.zig      # 公共入口：re-export + selectBackend 编排 + 全部 test 块
  backend.zig   # BackendType / SelectionMode / heuristic
  buffer.zig    # DeviceBuffer(T)
  engine.zig    # ComputeEngine(comptime bt) + scalar/simd 内核
  bench.zig     # nowNs / timeAdd / pickBest
  main.zig      # CLI demo（手动或智能选后端、跑、打印耗时表）
```

依赖方向（**无环**）：
- `root.zig` → {backend, buffer, engine, bench}
- `bench.zig` → {engine, buffer}
- `engine.zig` → backend
- `buffer.zig` → backend

---

## backend.zig

```zig
const std = @import("std");

pub const BackendType = enum {
    cpu_scalar,
    cpu_simd,
    gpu_webgpu,
    gpu_cuda,

    pub fn name(self: BackendType) []const u8 {
        return switch (self) {
            .cpu_scalar => "cpu_scalar",
            .cpu_simd => "cpu_simd",
            .gpu_webgpu => "gpu_webgpu",
            .gpu_cuda => "gpu_cuda",
        };
    }

    /// 本 demo 真正实现的是两个 CPU 后端。
    pub fn isImplemented(self: BackendType) bool {
        return switch (self) {
            .cpu_scalar, .cpu_simd => true,
            .gpu_webgpu, .gpu_cuda => false,
        };
    }
};

pub const SelectionMode = enum { manual, heuristic, benchmark };

/// 启发式：逐元素内核在 N 足够大时 SIMD 更有优势。
pub const simd_threshold: usize = 1024;
pub fn heuristic(size: usize) BackendType {
    return if (size >= simd_threshold) .cpu_simd else .cpu_scalar;
}

test "backend enum basics" {
    try std.testing.expectEqualStrings("cpu_simd", BackendType.cpu_simd.name());
    try std.testing.expect(BackendType.cpu_scalar.isImplemented());
    try std.testing.expect(!BackendType.gpu_cuda.isImplemented());
    try std.testing.expectEqual(BackendType.cpu_simd, heuristic(4096));
    try std.testing.expectEqual(BackendType.cpu_scalar, heuristic(128));
}
```

## buffer.zig

```zig
const std = @import("std");
const backend = @import("backend.zig");
const BackendType = backend.BackendType;

/// 统一数据容器。CPU 后端只走 cpu_ptr；GPU 后端预留 gpu_handle 与 to_device/to_host。
pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        backend: BackendType,
        cpu_ptr: []T,
        gpu_handle: u64 = 0,
        size: usize,

        pub fn init(allocator: std.mem.Allocator, backend: BackendType, size: usize) !Self {
            const mem = try allocator.alloc(T, size);
            return .{ .allocator = allocator, .backend = backend, .cpu_ptr = mem, .size = size };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.cpu_ptr);
            // GPU 后端在此释放显存（本 demo 未实现）。
        }

        /// CPU->设备。CPU 后端 no-op；GPU 后端 stub。
        pub fn toDevice(self: *Self) void {
            _ = self;
        }
        /// 设备->CPU。CPU 后端 no-op；GPU 后端 stub。
        pub fn toHost(self: *Self) void {
            _ = self;
        }
    };
}

test "buffer device buffer roundtrip" {
    const gpa = std.testing.allocator;
    var buf = try DeviceBuffer(f32).init(gpa, .cpu_simd, 16);
    defer buf.deinit();
    try std.testing.expectEqual(@as(usize, 16), buf.size);
    try std.testing.expectEqual(@as(usize, 16), buf.cpu_ptr.len);
    buf.toDevice();
    buf.toHost();
}
```

## engine.zig

```zig
const std = @import("std");
const backend = @import("backend.zig");
const BackendType = backend.BackendType;

/// 基于 comptime 的静态派发引擎。选择 .cpu_simd 时生成的机器码只含 SIMD 循环。
pub fn ComputeEngine(comptime bt: BackendType) type {
    return struct {
        pub const backend: BackendType = bt;

        pub fn name() []const u8 {
            return bt.name();
        }

        /// out[i] = a[i] + b[i]
        pub fn add(comptime T: type, out: []T, a: []const T, b: []const T) void {
            std.debug.assert(a.len == b.len and b.len == out.len);
            switch (bt) {
                .cpu_scalar => addScalar(T, out, a, b),
                .cpu_simd => addSimd(T, out, a, b),
                .gpu_webgpu, .gpu_cuda => @panic("computeAccel: gpu backend not implemented in minimal demo"),
            }
        }

        /// out[i] = alpha * x[i] + y[i]
        pub fn saxpy(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
            std.debug.assert(x.len == y.len and y.len == out.len);
            switch (bt) {
                .cpu_scalar => saxpyScalar(T, alpha, out, x, y),
                .cpu_simd => saxpySimd(T, alpha, out, x, y),
                .gpu_webgpu, .gpu_cuda => @panic("computeAccel: gpu backend not implemented in minimal demo"),
            }
        }
    };
}

// ---- 内核实现（可放在文件底部，为 file-private fn）----

fn addScalar(comptime T: type, out: []T, a: []const T, b: []const T) void {
    for (0..a.len) |i| out[i] = a[i] + b[i];
}

fn addSimd(comptime T: type, out: []T, a: []const T, b: []const T) void {
    const V = @Vector(8, T);
    var i: usize = 0;
    const chunks = a.len / 8;
    while (i < chunks * 8) : (i += 8) {
        const va: V = a[i..][0..8].*;
        const vb: V = b[i..][0..8].*;
        const vr = va + vb;
        @memcpy(out[i..][0..8], &@as([8]T, vr));
    }
    while (i < a.len) : (i += 1) out[i] = a[i] + b[i];
}

fn saxpyScalar(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
    for (0..x.len) |i| out[i] = alpha * x[i] + y[i];
}

fn saxpySimd(comptime T: type, alpha: T, out: []T, x: []const T, y: []const T) void {
    const V = @Vector(8, T);
    const va: V = @splat(alpha);
    var i: usize = 0;
    const chunks = x.len / 8;
    while (i < chunks * 8) : (i += 8) {
        const vx: V = x[i..][0..8].*;
        const vy: V = y[i..][0..8].*;
        const vr = va * vx + vy;
        @memcpy(out[i..][0..8], &@as([8]T, vr));
    }
    while (i < x.len) : (i += 1) out[i] = alpha * x[i] + y[i];
}

test "engine add correctness across backends" {
    const n = 1009; // 非 8 的倍数，验证尾部处理
    const a = [_]f32{2.0} ** n;
    const b = [_]f32{3.0} ** n;
    var outs: [n]f32 = undefined;
    var outv: [n]f32 = undefined;

    ComputeEngine(.cpu_scalar).add(f32, &outs, &a, &b);
    ComputeEngine(.cpu_simd).add(f32, &outv, &a, &b);
    try std.testing.expectEqualSlices(f32, &outs, &outv);
    try std.testing.expectEqual(@as(f32, 5.0), outs[0]);
    try std.testing.expectEqual(@as(f32, 5.0), outs[n - 1]);
}

test "engine saxpy correctness across backends" {
    const n = 1027;
    const x = [_]f32{2.0} ** n;
    const y = [_]f32{1.0} ** n;
    var outs: [n]f32 = undefined;
    var outv: [n]f32 = undefined;

    ComputeEngine(.cpu_scalar).saxpy(f32, 3.0, &outs, &x, &y); // 3*2+1 = 7
    ComputeEngine(.cpu_simd).saxpy(f32, 3.0, &outv, &x, &y);
    try std.testing.expectEqualSlices(f32, &outs, &outv);
    try std.testing.expectEqual(@as(f32, 7.0), outs[0]);
    try std.testing.expectEqual(@as(f32, 7.0), outs[n - 1]);
}
```

## bench.zig

```zig
const std = @import("std");
const backend = @import("backend.zig");
const buffer_mod = @import("buffer.zig");
const engine = @import("engine.zig");
const BackendType = backend.BackendType;
const DeviceBuffer = buffer_mod.DeviceBuffer;
const ComputeEngine = engine.ComputeEngine;

/// 单调时钟（nanosecond），Zig 0.16 用 posix.clock_gettime。
pub fn nowNs() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

/// 对固定后端跑 iters 次 add，返回总耗时（ns）。
pub fn timeAdd(comptime T: type, comptime bt: BackendType, out: []T, a: []const T, b: []const T, iters: usize) u64 {
    const t0 = nowNs();
    for (0..iters) |_| ComputeEngine(bt).add(T, out, a, b);
    const t1 = nowNs();
    return @intCast(@max(0, t1 - t0));
}

/// 运行时 bench 每个可用后端，返回最快的。
pub fn pickBest(allocator: std.mem.Allocator, comptime T: type, size: usize, iters: usize) !BackendType {
    var a = try allocator.alloc(T, size);
    defer allocator.free(a);
    var b = try allocator.alloc(T, size);
    defer allocator.free(b);
    var out = try allocator.alloc(T, size);
    defer allocator.free(out);
    @memset(a, 2.0);
    @memset(b, 3.0);

    var best: ?BackendType = null;
    var best_ns: u64 = std.math.maxInt(u64);
    inline for (BackendType) |bt| {
        if (!bt.isImplemented()) continue;
        const ns = timeAdd(T, bt, out, a, b, iters);
        if (ns < best_ns) {
            best_ns = ns;
            best = bt;
        }
    }
    return best orelse error.NoBackend;
}
```

## root.zig（公共入口 + 编排 + 全部 test）

```zig
const std = @import("std");
const backend = @import("backend.zig");
const buffer = @import("buffer.zig");
const engine = @import("engine.zig");
const bench_mod = @import("bench.zig");

pub const BackendType = backend.BackendType;
pub const SelectionMode = backend.SelectionMode;
pub const DeviceBuffer = buffer.DeviceBuffer;
pub const ComputeEngine = engine.ComputeEngine;
pub const bench = bench_mod;

/// 选择逻辑编排：manual / heuristic / benchmark。
pub fn selectBackend(
    allocator: std.mem.Allocator,
    mode: SelectionMode,
    manual_choice: BackendType,
    comptime T: type,
    size: usize,
    iters: usize,
) !BackendType {
    return switch (mode) {
        .manual => manual_choice,
        .heuristic => backend.heuristic(size),
        .benchmark => try bench_mod.pickBest(allocator, T, size, iters),
    };
}

// root.zig 中：下面的子文件被引用，它们的 test 块才被 zig build test 纳入。
test {
    _ = backend;
    _ = buffer;
    _ = engine;
    _ = bench_mod;
}

// ===== 关键 test：性能对比（scalar vs simd）=====
test "perf: simd beats scalar on large arrays" {
    const builtin = @import("builtin");
    const size: usize = 1 << 20; // 1,048,576
    const iters: usize = 20;

    const gpa = std.testing.allocator;
    var a = try gpa.alloc(f32, size);
    defer gpa.free(a);
    var b = try gpa.alloc(f32, size);
    defer gpa.free(b);
    var out_scalar = try gpa.alloc(f32, size);
    defer gpa.free(out_scalar);
    var out_simd = try gpa.alloc(f32, size);
    defer gpa.free(out_simd);

    @memset(a, 2.0);
    @memset(b, 3.0);

    // 正确性：两后端结果必须一致
    ComputeEngine(.cpu_scalar).add(f32, out_scalar, a, b);
    ComputeEngine(.cpu_simd).add(f32, out_simd, a, b);
    try std.testing.expectEqualSlices(f32, out_scalar, out_simd);
    try std.testing.expectEqual(@as(f32, 5.0), out_scalar[0]);
    try std.testing.expectEqual(@as(f32, 5.0), out_scalar[size - 1]);

    const t_scalar = bench_mod.timeAdd(f32, .cpu_scalar, out_scalar, a, b, iters);
    const t_simd = bench_mod.timeAdd(f32, .cpu_simd, out_simd, a, b, iters);
    const speedup = @as(f64, @floatFromInt(t_scalar)) / @as(f64, @floatFromInt(t_simd));

    std.debug.print("\n[perf perf] size={} iters={} scalar={}ns simd={}ns speedup={d:.2}x\n", .{ size, iters, t_scalar, t_simd, speedup });

    // ReleaseFast 下 SIMD 必须显著更快；Debug 下只打印并保证结果正确。
    if (builtin.mode == .ReleaseFast) {
        try std.testing.expect(t_simd < t_scalar);
        try std.testing.expect(speedup > 1.0);
    }
}

// 其它 test 块分散在 submodule，root 引用即可。
```

## main.zig（CLI demo，Task 3）

`pub fn main(init: std.process.Init) !void`，读 `init.minimal.args.toSlice(arena)` 与 `init.arena.allocator()`。

支持参数：
- `--backend <name>`：手动（cpu_scalar / cpu_simd / gpu_webgpu / gpu_cuda）。
- `--auto`：运行时 benchmark 选最快后端。
- `--heuristic`：按规模启发式。
- 缺省：heuristic。
- 可选 `--size <n>`、`--iters <n>`。

行为：
1. 分配 DeviceBuffer(f32)，填充 a=2.0、b=3.0。
2. 用 `selectBackend` 选出后端。
3. 打印选择结果与理由。
4. 跑 `ComputeEngine(chosen).add` 一次验证 + 计时。
5. 打印一张对比表：scalar 总耗时 vs simd 总耗时 vs 加速比（用 `bench.timeAdd` 对两种已实现后端都测一遍）。
6. 打印结果抽样（前 4 个元素）。

> 让 demo 能直接演示「手动指定 vs 智能选择」的差异。

---
**验收**：
- `zig build` 成功。
- `zig build test -Doptimize=ReleaseFast` 全绿（含 perf 测试，simd > scalar）。
- `zig build run -- --auto --size 1048576` 打印选择结果与耗时对比表。
