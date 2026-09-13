# handoff — T3：能力探测 / 诚实回退 / 选择与 bench 集成

> 当前任务文档。T1（native wgpu-native）见 commit `456e327`，T2（浏览器 wasm）见 `e522852`。
> 上游研究 `docs/gpu-backend-research.md`，依赖 `deps/README.md`。
> 这两个后端都已实测跑通，**不要重做**。T3 只做"选择与回退语义"，不改 kernel 与绑定。

## 0. 一句话

现在的 GPU 后端**能跑，但选择逻辑会说谎**：`ComputeEngine(.gpu_webgpu)` 失败时静默
`catch` 回退 CPU SIMD（`src/engine.zig`），于是 `--auto` 可能把"实际跑在 CPU 上的
gpu_webgpu"选成最优后端。T3 要把这条路修成：**探测 → 明确标注 → 诚实回退**。

## 1. 现状（先读代码确认，别凭猜）

| 位置 | 现状 | 问题 |
|---|---|---|
| `src/engine.zig` | `.gpu_webgpu => gpu_pipeline.add(...) catch addSimd(...)` | 静默回退：调用方无法区分"GPU 跑了"还是"回退跑了" |
| `src/bench.zig::pickBest` | `inline for` 里包含 `gpu_webgpu`，用 `timeAdd` 测 | 无能力探测；且测的是"GPU 或静默 CPU 回退"的混合体 |
| `src/backend.zig::heuristic` | 只看 size，`gpu_webgpu` 永不入选 | 对 GPU 无感知（当前这反而是"安全"的默认） |
| `src/main.zig` | `--auto`/`--heuristic`/`--backend` 三条路 | `--auto` 结果可能被静默回退污染 |
| `src/gpu/context.zig` | `GpuContext.init()` 会真的建 instance/adapter/device | 可作为探测入口，但每次调用都重建，需要"探测一次 + 缓存" |

**实测事实（很重要，别假设 GPU 一定更快）**：本机 AMD Vega iGPU 上，`add`
端到端 GPU 4.652 GB/s vs cpu_simd 14.676 GB/s（输），steady-state `gpu_batch`
16.249 GB/s vs 14.676 GB/s（略赢）。所以 **heuristic 默认不该在 `1<<20` 这种规模
选 GPU**；`--auto` 若选 GPU，必须是实测赢了才选，且要说明测的是哪种口径。

## 2. 交付物

1. **能力探测（一次 + 缓存）**
   - `GpuContext.probe() -> ProbeResult`（或 `isAvailable()`），内部只初始化一次并缓存结果
     （成功/失败原因），且**线程安全**（用 `std.once` 或原子，本机单线程但别留坑）。
   - 失败要携带原因（无 instance / 无 adapter / device 创建失败），不要只返回 bool。
2. **消除静默回退的误导**
   - 让调用方能知道"这次到底谁跑了"：例如 `gpu_pipeline.addEx(...) -> !void`，
     `ComputeEngine(.gpu_webgpu).add` 保留回退但把回退原因记到可查询处
     （如 `gpu.lastFallbackReason()`），CLI 打印 `gpu_webgpu (fell back: ...)`。
   - **不允许**在"选了 GPU 却回退"时打印成普通 GPU 结果。
3. **选择逻辑**
   - `heuristic(size)`：默认仍只选 CPU；仅在"探测可用 **且** size ≥ 阈值"时考虑 GPU，
     阈值要基于本仓库的实测数据（写成常量 + 注释说明数据来源）。
   - `bench.pickBest`：GPU 只有在 probe 成功时才纳入；测量口径统一（要么都端到端，
     要么对 GPU 单独标注 batched 并在返回值/输出里区分），**不要**拿 batched 数字
     去和 CPU 端到端比而宣称 GPU 更快。
4. **测试**（`zig build test` 里）
   - probe 失败路径：伪造/无 GPU 情况下 `pickBest` 不返回 `gpu_webgpu`。
   - heuristic 在阈值两侧的行为。
   - 回退路径：GPU 不可用时不 panic、不改变 CPU 结果。
   - 保持现有 8 个测试全绿（CPU 行为、GPU 正确性测试的判定不能放松）。
5. **README**：更新"选择逻辑"与"回退语义"两节，写清什么时候会选 GPU、怎么知道发生了回退。

## 3. 硬性约束

- **不要改 kernel、绑定、shader、wasm/emcc 相关**（T1/T2 已验收，改动会破坏已通过的验收）。
- 不要用 `wgpuInstanceWaitAny` / `wgpuDevicePoll` / SPIR-V / ASYNCIFY。
- 不要引入第三方依赖。
- 不要放宽既有断言来"让测试过"。
- **不要 kill 任何进程、不要 kill/ps 你的父进程**：你只需要改代码。
  如果你想看是否有构建在跑，用 `ls`/`git status`，不要动进程。
- 完成后 `git commit`（中文，含实测输出）。

## 4. 验收标准（自己实测并贴证据）

1. `zig build test --summary all` → 全绿（含新增测试）。
2. `zig build run -- --auto --size 1048576` → 输出不谎报：若选 `gpu_webgpu`，
   必须能说明它实测更快；若选 `cpu_simd`，也要能解释（当前本机就该是这种情况）。
3. 人为制造 GPU 不可用的路径（例如临时用 `VK_ICD_FILENAMES=/nonexistent.json`
   或在测试里注入失败），验证：
   - 不 panic；
   - 结果仍正确（回退 CPU）；
   - 输出明确标注发生了回退。
4. `zig build run -- --backend gpu_webgpu --size 1048576 --iters 20` 仍正常（T1 验收不回归）。
5. `git commit`。

## 5. 命令

```bash
zig build test --summary all
zig build run -- --auto --size 1048576
zig build run -- --heuristic --size 1048576
zig build run -- --backend gpu_webgpu --size 1048576 --iters 20
VK_ICD_FILENAMES=/nonexistent.json zig build run -- --backend gpu_webgpu --size 1048576
tools/check_abi_drift.sh        # 只读检查，不改绑定
```

> 注：`zig build` 输出里若出现 `failed command: ...`，是 Zig 0.16 对"子命令往 stderr
> 写过东西"的前缀噪音（emcc 的 clang 版本 warning 也会触发）；以
> `Build Summary: ... success` 和退出码为准。
