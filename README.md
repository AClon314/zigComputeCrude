# computeAccel — 最小 demo（智能/手动选择计算后端）

一个 Zig **0.16.0** 的最小示例，演示「**手动或智能选择计算后端以加速计算**」。

核心架构遵循 `TODO.md` 的两层分离 + comptime 静态多态：
- **数据容器** `DeviceBuffer(T)`：统一分配/释放，预留 CPU↔GPU 同步接口。
- **计算引擎** `ComputeEngine(comptime BackendType)`：用 `comptime switch` 静态派发，编译期剥离无关分支 —— 选 `.cpu_simd` 时生成的机器码只含 SIMD 循环，零运行时开销。

## 支持的后端

| BackendType | 实现状态 |
|---|---|
| `cpu_scalar` | ✅ 标量循环（兜底） |
| `cpu_simd` | ✅ `@Vector(8, T)` SIMD |
| `gpu_webgpu` | ⬜ 枚举占位，调用即 panic（未接入） |
| `gpu_cuda` | ⬜ 枚举占位，调用即 panic（未接入） |

## 选择逻辑（`selectBackend`）

- `manual`：用户显式指定（`--backend <name>`）。
- `heuristic`：按规模启发式（`size >= 1024` → SIMD，否则 scalar；`--heuristic`）。
- `benchmark`：运行时对每个可用后端实测（`bench.pickBest`），选耗时最短者（`--auto`）。

## 构建与运行

```bash
zig build                      # 编译
zig build test -Doptimize=ReleaseFast   # 跑库测试（含 perf 对比），建议 ReleaseFast 才能看到 SIMD 优势
zig build test                 # Debug 下只跑正确性 + 打印对比（不强制断言加速比）

# 手动指定后端
zig build run -- --backend cpu_scalar --size 1048576 --iters 20
zig build run -- --backend cpu_simd  --size 1048576 --iters 20
# 智能（实测选最快）
zig build run -- --auto --size 1048576
# 启发式（按规模）
zig build run -- --heuristic --size 1048576      # 大 → simd
zig build run -- --heuristic --size 128          # 小 → scalar
# 占位后端
zig build run -- --backend gpu_cuda              # 打印「未接入」提示
```

## 关键 test：性能对比

`src/root.zig` 的 `test "perf: simd beats scalar on large arrays"`：
1. 对 `size=1<<20` 的数组分别跑 scalar / SIMD `add`。
2. **断言结果一致**（正确性）。
3. 用 `bench.timeAdd` 计时两者，打印 `total_ns` 与加速比。
4. `ReleaseFast` 下**断言 SIMD 严格更快**；Debug 下只打印对比（避免小输入/无优化下误判）。

本机实测（ReleaseFast，`size=1048576, iters=20`）：

```
backend      total_ns    throughput(GB/s)
cpu_scalar   49908866   3.362
cpu_simd     11312730   14.830
speedup (scalar/simd) = 4.41x
```

SIMD 约是标量的 **4.4x**。

## 设计要点

- **零运行时开销**：comptime 派发，GPU 分支在编译期被剥离。
- **扩展性**：新增后端只需在 `BackendType` 加枚举 + 在 `ComputeEngine` 的 `switch` 加分支，业务代码不动。
- **数据生命周期清晰**：`toDevice()`/`toHost()` 显式同步，避免隐藏拷贝。

## 结构

```
src/
  root.zig      # 公共入口 + selectBackend 编排 + 全部测试
  backend.zig   # BackendType / SelectionMode / heuristic
  buffer.zig    # DeviceBuffer(T)
  engine.zig    # ComputeEngine(comptime bt) + scalar/simd 内核
  bench.zig     # nowNs / timeAdd / pickBest（实测选优）
  main.zig      # CLI demo
SPEC.md         # 设计契约（API 事实来源）
```
