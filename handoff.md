# handoff — T4：修复超出 dispatch 上限的 GPU 路径（>65535 workgroups）

> T1 `456e327`（native）、T2 `e522852`（浏览器 wasm）、T3 `94d7712`（探测/诚实回退/选择）均已提交验收。
> 本任务修 T3 暴露出来的一个**真实功能缺陷**。

## 0. 问题（已实测定位到根因）

T3 的诚实回退把问题暴露出来了：**`size` 稍大一点 GPU 调用就 GpuError 并回退 CPU**。

我实测（本机 AMD Vega/RADV，`--backend gpu_webgpu`）：

| size | workgroups (= size/64) | 结果 |
|---|---|---|
| 4194240 | 65535 | ✅ 正常，3.354 GB/s |
| 4194304 | 65536 | ❌ `fell back: GpuError` |
| 8388608 | 131072 | ❌ `fell back: GpuError` |

根因：`src/gpu/pipeline.zig` 里只用了 1D dispatch：

```zig
wgpu.wgpuComputePassEncoderDispatchWorkgroups(pass, @intCast(workgroup_count), 1, 1);
```

而 WebGPU 的 `maxComputeWorkgroupsPerDimension` 是 **65535**，超过就 validation error。
代码里**完全没有查询 device limits**（`rg 'Limits|maxCompute' src/gpu/` 为空）。

同时 `src/backend.zig` 的 heuristic GPU 闸门（T3 加的 `1<<22` = 4194304）正好落在这个
失败区间里 —— 所以 `--heuristic --size 8388608` 会选一个必然回退的 GPU 路径。
这是 T3 遗留的坑，T4 要一起修掉。

## 1. 交付物

### A. 让 dispatch 支持任意 size（核心）

两条路，选一条并写清理由（注释里说明）：

1. **2D grid（推荐）**：`x = min(limit, groups)`，`y = ceil(groups / x)`，
   shader 用 `@builtin(num_workgroups)` 还原线性下标：
   `gid = global_invocation_id.x + global_invocation_id.y * (num_workgroups.x * workgroup_size_x)`
   （`workgroup_size_y = 1` 时 `global_invocation_id.y` 就是 y 方向的 workgroup 序号）。
   注意 `add.wgsl` / `saxpy.wgsl` 都要改（同一套写法），并保持越界保护的语义不变。
2. **多次 1D dispatch**：把 `groups` 切成 ≤65535 的块，每块 dispatch 一次，shader 用
   参数 buffer 里的 `offset` 偏移下标。需要改 params uniform（注意 wgpu-native v29 已把
   push constants 改名为 immediates，**不要用 immediates/push constants**，用 uniform buffer）。

要求：**不改 kernel 语义**、不改变已有正确性断言；`workgroup_size(64)` 可保留。

### B. limits 感知的能力探测

- 用 `wgpuAdapterGetLimits` / `wgpuDeviceGetLimits`（都在已绑定的头文件里，符号名先在
  `vendor/wgpu-native/include/webgpu/webgpu.h` 与 `.em-cache/.../emdawnwebgpu_pkg/webgpu/include/webgpu/webgpu.h`
  里核对）读取并缓存：
  - `maxComputeWorkgroupsPerDimension`
  - `maxStorageBufferBindingSize`
  - `maxBufferSize`
- `GpuContext.probe()` 结果里带上这些 limits；
- 提供 `GpuContext.canRun(n_bytes, groups) -> bool`（或等价 API），让选择逻辑能问
  "这个 size 的 GPU 路径可行吗"。

### C. 选择逻辑

- `heuristic` 的 GPU 闸门不能只看 size，必须**同时**满足：probe 成功 + size/字节数在 limits 内。
  修完后 `--heuristic --size 8388608` 必须走通（GPU 真跑，不退化成 GpuError 回退）。
- `bench.pickBest` / `--auto` 沿用 T3 语义（真实端到端胜出才选 GPU），但要保证
  大 size 下 GPU 候选是"真的能跑"的，而不是必然失败后回退。

### D. 测试

- 新增：`size` 刚超过 65535 workgroups 的 GPU 正确性测试（例如 `1<<22`，即 65536 groups），
  与 `cpu_scalar`/`cpu_simd` 逐元素一致，**且断言没有发生 fallback**。
- 新增：probe/limits 相关单元测试（例如 `canRun` 边界）。
- 保持现有 11 个测试全绿。

### E. README

更新"限制与已知边界"：workgroup 上限怎么处理、大 size 表现、limits 从哪来。

## 2. 硬性约束

- 不要破坏 T1/T2 的验收：`zig build test` 全绿、`zig build wasm` 成功、
  `--backend gpu_webgpu --size 1048576` 正常。
- 不要用 `wgpuInstanceWaitAny` / `wgpuDevicePoll` / SPIR-V / ASYNCIFY / push constants(immediates)。
- **不许 kill 任何进程**（尤其不要动 pi 进程）。
- 不要放宽既有断言来"让测试过"；不要伪造实测结论。
- 完成后 `git commit`（中文，含实测输出）。

## 3. 验收标准

1. `zig build test --summary all` → 全绿（含新增测试）。
2. `--backend gpu_webgpu` 在 `size = 4194304` 与 `8388608` 上**不再 fallback**，
   结果 `5.0`，并打印吞吐。
3. `--heuristic --size 8388608` → 选中 GPU 且真的跑在 GPU 上（不退化成 GpuError）。
4. `--backend gpu_webgpu --size 1048576` 仍正常（T1 不回归）。
5. `zig build wasm` 成功（T2 不回归），并说明浏览器端 shader 是否同步改动。
6. `tools/check_abi_drift.sh` → OK。

## 4. 命令

```bash
zig build test --summary all
zig build run -- --backend gpu_webgpu --size 4194304 --iters 5
zig build run -- --backend gpu_webgpu --size 8388608 --iters 5
zig build run -- --heuristic --size 8388608
zig build run -- --auto --size 1048576
zig build wasm
tools/check_abi_drift.sh
```
