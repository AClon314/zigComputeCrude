# handoff — T5/T6/T7：Step 3 新 kernel（GEMM simple/tiled + reduce）

> 路线图见 `handoff-3steps.md`（不改）。上游已完成：T4（>65535 workgroups 的 2D dispatch）
> `eff5d3e`、Step 2（ABI 漂移检查）`fda10dc`。
> 本次任务按路线图建议顺序做 **Step 3**：GEMM simple → GEMM tiled → reduce。
> **Step 1（显存常驻 + 链式提交）仍未开始**，是路线图里剩下的最后一块。

## 0. 状态

| 项 | 状态 |
|---|---|
| GEMM simple + tiled kernel、CPU 参考对拍、GFLOP/s 基准 | ✅ 完成 |
| reduce（sum/max，两趟归并） | ✅ 完成 |
| CLI（`--kernel gemm|reduce`）与实测数字 | ✅ 完成，数字见 §2 |
| `zig build test` | ✅ 24/24 绿（原 15 + 新增 9）；GPU 测试实际执行、无 skip |
| `zig build wasm` | ✅ 改动后重跑成功（新 kernel 只在 native 路径，wasm 仍只跑 add） |
| `tools/check_abi_drift.sh` | ✅ OK（未改绑定） |
| README（新 kernel 说明 + 实测数字 + 踩坑） | ✅ 已更新 |
| 1024³ 多次迭代的 wgpu-native abort | ✅ 已修复，根因见 §3 |
| `git commit` | ✅ `9042bbd`（工作区干净） |

## 1. 交付内容

代码：

- `src/gpu/shaders/gemm_simple.wgsl`：每个 invocation 算一个输出元素，只用全局内存；
  线性下标从 2D grid 还原（与 add/saxpy 同一套 `num_workgroups` 写法）。
- `src/gpu/shaders/gemm_tiled.wgsl`：64 线程/workgroup，16×16 输出 tile，A/B block
  进 `var<workgroup>`、`workgroupBarrier()` 分隔，每线程 2×2 micro-tile，BK=16，
  边界零填充（支持任意 m/k/n）。
- `src/gpu/shaders/reduce.wgsl`：`sum_main` / `max_main` 两个 entry point；
  pass 1 每个 workgroup 用 grid-stride 归约出 partial，pass 2 用 1 个 workgroup
  归约 partials；workgroup 内 64 个累加器经共享内存 + barrier 树形归并。
- `src/gpu/gemm.zig`：`Variant = {simple, tiled}`；`canRun(limits, m,k,n,variant)`
  同检 3 个 buffer 与 dispatch grid；cache 按 `(device, shape)` 复用（两个 variant
  共用一套 buffer/bind group layout）；`runWithContext` / `runBatchedWithContext`
  / `gemm` / `gemmBatched`；CPU 参考 `referenceScalar`、`referenceSimd`
  （4×8 寄存器分块）、`maxAbsDiff`。
- `src/gpu/reduce.zig`：`Op = {sum, max}`，两趟 dispatch、两个 bind group（同一
  layout）；cache 按 `(device, n, buckets)`；CPU 参考 scalar + 8-lane SIMD。
- `src/gpu/context.zig`：新增 `GpuLimits.workgroupGrid()` + `WorkgroupGrid`；
  `GpuContext.readBuffer()`（mapAsync + pump + getMappedRange + memcpy + unmap，
  无 WaitAny/DevicePoll）；`waitFor` 改为墙钟超时（见 §3）。
- `src/gpu/pipeline.zig`：`mapRead` 在等待超时时取消挂起 mapping（同一坑的另一条路径）。
- `src/bench.zig`：`timeGemmCpu/timeGemm/timeGemmBatched`、
  `timeReduceCpu/timeReduce/timeReduceBatched`（GPU 错误不吞成 CPU 时间；
  CPU reduce 计时每轮扰动一个元素，防止 ReleaseFast 把纯归约提到循环外）。
- `src/root.zig`：导出 `computeAccel.gemm`、`computeAccel.reduce` 并纳入 test。
- `src/main.zig`：`--kernel add|gemm|reduce`（默认 add）、`--m/--k/--n`、
  `--variant simple|tiled|both`、`--op sum|max`；GEMM 打印 GFLOP/s、reduce
  打印 GB/s；每条 GPU 结果都与 `cpu_simd` 逐元素对拍并打印 `max|diff|`；
  各 kernel 默认 iters：add=20 / gemm=5 / reduce=10。

测试（新增 9 个，`zig build test` 24/24）：

- GEMM：CPU scalar/simd 在 7 种边角 shape 一致；GPU simple/tiled 在
  `1x1x1 / 2x3x5 / 13x17x19 / 64³ / 100x90x80` 与 CPU 一致（容差 1e-4 相对，
  实测 max|diff|=0）；`canRun` 边界；**长 GPU 工作的 readback 回归测试**（§3）。
- reduce：CPU scalar/simd 一致；GPU sum/max 在 `1/63/64/65/1023/4099/1<<16`
  与 CPU 一致（max 精确、sum 在容差内）；`1<<20` 个 1.0 求和**位精确**；
  `canRun` 边界。

## 2. 本机实测（Ryzen 5 5600H + Radeon Vega iGPU，RADV/Vulkan）

所有 GPU 行均与 `cpu_simd` 对拍通过（`max|diff| = 0`）。CPU 对比对象是自写的
4×8 寄存器分块 SIMD，**不是 BLAS**。

ReleaseFast GEMM（端到端含上传/回读；batch = 上传一次 + iters 次 dispatch + 回读一次）：

| workload | cpu_simd | gpu_simple e2e | gpu_tiled e2e | gpu_tiled batch |
|---|---|---|---|---|
| 512³, iters=10 | 35.4 GFLOP/s | 71.1 (2.0x) | **164.8 (4.7x)** | 219.0 (6.2x) |
| 1024³, iters=5 | 22.4 | 27.0 (1.2x) | **188.6 (8.4x)** | 223.7 (10.0x) |
| 2048³, iters=3（tiled） | 12.7 | — | **212.8 (16.7x)** | 230.0 (18.1x) |

ReleaseFast reduce（GB/s 按输入字节）：

| workload | cpu_simd | gpu e2e | gpu batch |
|---|---|---|---|
| 4M sum, iters=10 | 30.5 GB/s | 5.7 (0.19x) | 13.1 (0.43x) |
| 4M max, iters=10 | 28.2 | 6.0 | 14.1 |
| 16M sum, iters=10 | 25.8 | 7.1 (0.27x) | 25.5 (≈1.0x) |

结论：GEMM 是「计算量压过搬运量」的典型，tiled 从 512³ 起端到端即可胜出，2048³
到 16.7x；simple 只在 512³ 靠 L2 命中勉强赢，之后被 tiled 拉开 —— 分块优化的价值
被量化出来了。reduce 是纯流式读，在这台共享内存的 iGPU 上没有优势：端到端输在
「把输入写进显存」这一趟，batch 在 16M 时也只与 CPU SIMD 打平（双方都到内存带宽
上限）。reduce 的真正场景是数据本就常驻显存（Step 1 的链式 workload）。

## 3. 已解决的坑：1024³ 多迭代时 wgpu-native abort（"Buffer is still mapped"）

### 现象

ReleaseFast、1024³、`--iters 5`（默认）：

```text
thread '<unnamed>' panicked:
Error in wgpuQueueSubmit: Validation Error
Caused by: Buffer with '' label is still mapped
# SIGABRT（exit 134；wgpu-native 的 fatal validation error，error scope 捕不到）
```

iters 1/2/3 正常；`--variant tiled` 正常；`--variant simple --iters 5` 的 batch
先报可捕获的 `GpuError`。

### 根因

不是 wgpu 的 bug，是 `GpuContext.waitFor` 用**固定 100000 次 ProcessEvents**
当超时：泵的次数与排队中的 GPU 工作成正比（当时实测一次 4MB map 需要 ~31776 次
泵 ≈ 76ms，而 5 连发 simple 批处理要 ~380ms），所以设备还在跑时等待就提前到期，
`readBuffer` 返回错误但 mapping 停留在 `Waiting`；下一次 `wgpuQueueSubmit` 的
command buffer 正好用到这个 staging buffer，wgpu-core 的
`validate_command_buffer` 发现 `map_state != Idle` → `BufferStillMapped`，
而 wgpu-native 对 submit 错误直接 `handle_error_fatal` → abort。

### 修法

- `waitFor` / `waitForInitialization` 改为 **30s 墙钟预算**（不是工作量的估算，
  只防设备真卡死）；
- `readBuffer`（以及 `pipeline.zig` 的 `mapRead`）在等待超时时调用
  `wgpuBufferUnmap` **取消挂起的 mapping**，让 buffer 回到 Idle，后续 submit 不再中招；
- 回归测试：`gemm batched long gpu work does not expire the readback wait`
  （1024³ simple 批处理 6 连发 + 紧接着 tiled 单发；旧实现下该测试失败）。

教训：**读回等待不能拿迭代次数当超时**；这个坑对 add 的 `gpu_batch`（大 size + 多
iters）同样存在，现已同时覆盖。

## 4. 命令速查

```bash
zig build test --summary all                    # 24/24（含新 kernel 与回归测试）
zig build run -- --kernel gemm --m 512 --k 512 --n 512
zig build run -Doptimize=ReleaseFast -- --kernel gemm --m 1024 --k 1024 --n 1024 --iters 5
zig build run -- --kernel gemm --m 2048 --k 2048 --n 2048 --variant tiled --iters 3
zig build run -- --kernel reduce --size 4194304 --op sum
zig build run -- --kernel reduce --size 16777216 --op max
zig build wasm                                  # 已重跑成功
tools/check_abi_drift.sh                        # 未改绑定，OK
```

## 5. 后续建议（路线图剩余项）

- **Step 1（T8，大工程）**：显存常驻 buffer + 显式 `toDevice/toHost` + 多 kernel
  链式提交。现在 GEMM/reduce 的 cache 已经是「按 device/shape 常驻」，但 API 仍是
  「每次调用一次往返」；把 `gemm`/`reduce`/`add` 串成一条 command buffer 只提交
  一次，能直接验证「唯一那次搬运被链长摊薄」。
- reduce 若要真正赢，应在 Step 1 之后用于「输入已常驻显存」的链路，而不是拿它和
  CPU 做纯流式读的对比。
- 浏览器端目前只跑 add；新 kernel 的 WGSL 与绑定是共享的，移植是机械工作。
