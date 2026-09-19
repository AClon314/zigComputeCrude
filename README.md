# computeAccel — 可移植计算内核与常驻链式运行时（Zig + WebGPU）

一个 Zig 0.16 库/演示：CPU SIMD 与 WebGPU 计算共用一套内核定义，
**同一份绑定的 C ABI 与同一份 WGSL 同时编译到 native（wgpu-native）与浏览器（emdawnwebgpu）**。
除了逐元素/GEMM/归约内核，还提供 M0 运行时：**常驻显存 buffer + 多 dispatch 一次提交、
一次回读**，用于把"每步一次搬运"的工作负载变成"搬一次算很多步"。

- 使用/安装：本文档（面向调用者）
- 开发与贡献规则：`AGENTS.md`
- 架构与后续路线（Blender 节点系统迁移评估）：`docs/node-system-migration.md`
- 依赖说明：`deps/README.md`
- GPU 选型与生态核查：`docs/gpu-backend-research.md`、`docs/zig-gpu-spike.md`（Zig 当 kernel 语言的实测否决）
- 性能分析与 comptime 判据（工具 + A/B 实测）：`docs/perf-tooling-and-comptime.md`

---

## 能力现状

| 层           | 内容                                                                                                     | 状态                                  |
| ------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 后端         | `cpu_scalar`、`cpu_simd`（`@Vector`，目标自适应位宽 + `@mulAdd`）、`gpu_webgpu`                          | ✅                                    |
| 内核         | `add`、`saxpy`、`bias_add`（广播加）、`gemm`（simple/tiled）、`reduce`（sum/max，两趟归约）              | ✅                                    |
| 运行时（M0） | `Buffer`（常驻 + 脏标记）、`Kernel`（WGSL+entry+binding 描述）、`Chain`（多 dispatch/一次提交/一次回读） | ✅                                    |
| 选择逻辑     | `manual` / `heuristic`（阈值 + limits 闸门）/ `benchmark`（真实端到端实测）                              | ✅                                    |
| 能力探测     | `probe()` + limits + 诚实回退（失败原因可查询，绝不把 GPU 失败算成 GPU 时间）                            | ✅                                    |
| 未覆盖       | 纹理/采样器、indirect dispatch、atomics/scan/sort、f16/u32、BVH、多队列                                  | ⬜ 见 `docs/node-system-migration.md` |

约束（两端共同的底线）：只用 WGSL（浏览器拒绝 SPIR-V）、不用
`wgpuInstanceWaitAny`/`wgpuDevicePoll`、不用 push constants/immediates、
浏览器端不启用 ASYNCIFY（只能 `ProcessEvents` + rAF pump）。当前 GPU 路径只支持 `f32`。

---

## 安装

依赖：

- **Zig 0.16.0**（`zig version`）；
- **wgpu-native**（native 端）：`tools/fetch_deps.sh` 按 `deps/pins.env` 拉到 `vendor/wgpu-native/`；
- **Emscripten + emdawnwebgpu**（仅浏览器构建需要）：emcc 通过 remote port 自动解包到 `.em-cache/`。

```bash
tools/fetch_deps.sh                 # 拉 wgpu-native（预编译 .so + 头文件）
zig build                           # 产出 zig-out/bin/computeAccel 与 zig-out/lib/libwgpu_native.so
zig build test                      # 全部测试（GPU 不可用时相关测试自动 Skip）
zig build test -Doptimize=ReleaseFast
zig build wasm                      # 浏览器版：zig-out/webgpu/{computeAccel.js,.wasm,shell.html}
tools/check_abi_drift.sh            # 绑定 ABI 漂移检查（native vs emdawnwebgpu 头文件）
```

`zig build wasm` 后可用 `cd zig-out/webgpu && python3 -m http.server 8080`，
再用 Chrome 打开 `http://127.0.0.1:8080/shell.html`。

页面会**逐块测所有能被请求到的 GPU**：浏览器没有 `enumerateAdapters`，所以它用四种请求
（`auto` / `high-performance` / `low-power` / `forceFallbackAdapter`）枚举，去重后
**对每块卡跑同一份 WGSL add**（CPU SIMD 参考 + 位精确比较），并对照 JS 侧 `adapter.info`
与 C ABI 侧自己请求到的适配器身份（防止混合显卡上两边落到不同的卡）。
`?power=high-perf|low-power|fallback` 指定首轮用哪块卡。

实测（Chrome 154 + Vulkan，本机三块可用适配器）：两个 URL 变体下
`nvidia · ampere`（auto/high-performance）、`amd · gcn-5`（low-power）、
`google · swiftshader`（fallback）**全部 MATCH**，且 JS↔C ABI 身份逐块一致；
表里的 `GPU add` 是单次端到端耗时（含管线创建/上传/回读），只作量级参考，不是稳态基准。
**浏览器默认偏好按规范是 `low-power`**：不传 `powerPreference` 时拿到的可能是 iGPU，
这一点与 native 侧一致（见上面「adapter 消融」）。

---

## 模块（发布形态）

一个包（`build.zig.zon`）暴露两个 module，按需引入：

| module | 内容 | 链接依赖 |
|---|---|---|
| `computeAccel` | CPU 内核、GPU 后端（native wgpu-native / browser emdawnwebgpu）、runtime（Buffer/Kernel/Chain）、GEMM/reduce 等原语、能力探测与选择 | wgpu-native 可通过 `b.dependency(..., .{ .webgpu = false })` 关闭（CPU-only 消费者） |
| `computeAccel_spatial` | 与领域无关的空间原语（S1）：均匀网格索引（GPU build + 半径查询，与 CPU 参考逐元素对拍） | 无额外链接（复用 `computeAccel`） |

`computeAccel` 内的 `primitives/` 放领域无关的并行原语（当前：`scan`），
每个都带 CPU 参考实现与逐元素对拍测试。

```zig
// 消费者 build.zig
const accel = b.dependency("computeAccel", .{
    .target = target,
    .optimize = optimize,
    .webgpu = false, // CPU-only 消费者：不链接 wgpu-native（也不需要它存在）
});
exe.root_module.addImport("computeAccel", accel.module("computeAccel"));
exe.root_module.addImport("computeAccel_spatial", accel.module("computeAccel_spatial")); // 可选
```

完整可运行示例见 `examples/cpu_consumer/`（path 依赖 + `-Dwebgpu=false`），
`zig build consumer-check` 会在本仓库里构建并运行它，并断言二进制没有链接
wgpu-native。GPU 消费者需要提供 wgpu-native：仓库内默认用 `vendor/wgpu-native/lib`
（`tools/fetch_deps.sh` 拉取），发布形态下可用 `-Dwgpu-lib-dir=<目录>` 指向自己的
构建产物（published 包不包含 `vendor/`）。

Zig 是**惰性分析**：没被引用的声明（函数/类型/泛型实例/整个文件）不会被语义分析，
更不会进入产物——所以未使用的 module/内核零成本，不需要 TS 那种 `sideEffects` 注解。
仓库自带门禁验证这一点：

```bash
zig build tree-shake      # CPU-only 消费者的目标文件必须 0 个 wgpu/WGSL/spatial 符号
```

本机实测（ReleaseFast）：只调用 CPU add 的消费者对象 **10,792 B**（0 wgpu 符号、0 WGSL 文本）；
同一消费者加上 `accel.gpu.add` 后 **201,872 B**（42 个 wgpu 符号、1 份内嵌 WGSL）。
脚本：`tools/check_tree_shake.sh`。

## 快速开始（作为库使用）

### 1. 简单路径：选后端 + 跑内核

```zig
const accel = @import("computeAccel");
const n = 1 << 20;

// 手动或自动选择（benchmark 模式会真实跑端到端并选最快）
const backend = accel.selectBackend(allocator, .heuristic, .cpu_simd, f32, n, 20) catch .cpu_simd;

var out: [n]f32 = undefined;
accel.ComputeEngine(backend).add(f32, &out, &a, &b);

// GPU 内核（失败可用 lastFallbackReason() 查询原因）
accel.gpu.add(&out, &a, &b) catch {};             // add/saxpy
accel.gemm.gemm(.tiled, m, k, n, a, b, out) catch {};   // GEMM simple/tiled
accel.reduce.reduce(.sum, &scalar_out, input) catch {}; // reduce sum/max
```

### 2. 常驻 + 链式（M0 runtime，推荐用于多步工作负载）

```zig
const accel = @import("computeAccel");
const rt = accel.runtime;

const ctx = try rt.open();                       // 进程内单例设备；失败时原因在 accel.gpu.lastFallbackReason()

// 常驻 buffer：CPU -> 设备只上传一次，之后由脏标记决定是否重传
var x = try rt.Buffer.init(ctx, n * @sizeOf(f32),
    rt.buffer.storage_rw);                       // Storage | CopyDst | CopySrc
defer x.deinit(ctx);
try x.toDevice(ctx, std.mem.sliceAsBytes(host_x));

// 编译一个内核：WGSL + entry + binding 访问模式（write 用于脏标记）
var kernel = try rt.Kernel.init(ctx, wgsl_source, "main", &.{
    .{ .kind = .storage, .access = .read },
    .{ .kind = .storage, .access = .read },
    .{ .kind = .storage, .access = .write },
    .{ .kind = .uniform, .access = .read },
}, 64);
defer kernel.deinit();

const bind = try kernel.createBindGroup(&.{ &x, &y, &result, &params });
defer rt.releaseBindGroup(bind);
const grid = try kernel.gridLinear(n);           // 线性元素 -> 2D 展平 grid

// 多步链：N 个 dispatch 录进一条 command buffer，只提交一次、只回读一次
var chain = try rt.Chain.begin(ctx);
defer chain.deinit();                            // 未 submit 时的清理
for (0..steps) |_| {
    try chain.dispatch(&kernel, bind, &.{ &x, &y, &result, &params }, grid);
}
try chain.download(&result, std.mem.sliceAsBytes(host_out));
try chain.submit();                 // 提交并阻塞回读

// 或者异步：提交后先做 CPU 工作，再统一回读
// try chain.submitAsync();  // 不阻塞
// ... 准备下一批参数 / 提交更多 chain ...
// try chain.wait();         // 只在这里阻塞并映射结果
```

已验证的收益见下文「M0 消融实验」。设计原则：runtime 只做**常驻、链式、脏标记**三件事，
不引入未经验证的抽象（见 `AGENTS.md` 的"消融优先"）。

---

## 命令行

```bash
# 后端选择 demo（add）
zig build run -- --backend gpu_webgpu --size 1048576 --iters 20
zig build run -- --auto --size 1048576
zig build run -- --heuristic --size 8388608

# GEMM / reduce（输出 GFLOP/s、GB/s，并与 cpu_simd 逐元素对拍）
zig build run -- --kernel gemm --m 512 --k 512 --n 512 [--variant simple|tiled|both]
zig build run -- --kernel reduce --size 4194304 --op sum|max

# S1 空间索引：GPU 均匀网格 build + 半径查询（与 CPU 网格/暴力对拍）
zig build run -- --kernel spatial --points 262144 --queries 4096 --radius 2.0 --iters 3
# points 默认 65536、queries 默认 4096、radius 默认 2.0；点/查询都生成在 [0,64)^3

# M0 消融：常驻 + 链式（per_call / per_submit / chained × 链长）
zig build run -- --kernel chain --chain saxpy --size 4194304 --chain-lens 1,4,16,64 --iters 3
zig build run -- --kernel chain --chain pipeline --m 512 --k 512 --n 512 --chain-lens 1,4,16

# WebGPU adapter 选择（init-time 开关；默认 auto 不一定是独显）
zig build run -- --kernel gemm --m 1024 --k 1024 --n 1024 --adapter high-perf
zig build run -- --kernel chain --chain pipeline --chain-lens 64 --iters 10 --adapter auto
zig build run -- --kernel reduce --size 16777216 --adapter low-power

# 对拍判据档位（exact 会让 f32 累加类如预期 MISMATCH，这是设计而非回归）
zig build run -- --kernel chain --chain pipeline --chain-lens 16 --precision exact
zig build run -- --kernel reduce --size 16777216 --op sum --precision fast
```

`--kernel chain --chain pipeline` 跑的是 **GEMM → bias → reduce** 四段异质链
（三种不同内核、两种输出形状），用于验证多内核依赖顺序与"一次提交"。

`--precision exact|tolerant|fast` 选**对拍判据**（表在 `src/determinism.zig`）：离散量
（索引/计数/max）在任何档位下都精确相等，f32 单次舍入 / 累加类按档给量级。
判据在计时区**之外**，所以档位不影响测得的 kernel 时间（实测三档 gemm 512³ 为
202.8 / 202.6 / 203.5 GFLOP/s，属噪声）——"游戏要快"的收益来自 adapter 与链式，不是放宽判据。

`--adapter auto|high-perf|low-power` 是 **init-time** 开关，必须早于任何 GPU 调用；
每个偏好各自缓存一个 context（可以同一进程里跑两种 adapter 对比，见下面消融）。
库消费者用 `computeAccel.setAdapterSelection(.{ .preference = .high_performance })`
（或 `GpuContext.initWithAdapter`）；`probe().adapter_info.description()`
可拿到实际选中的卡名。浏览器端对应 `shell.html?power=high-perf`（同时转给 C ABI 侧）。

---

## 实测

环境：AMD Ryzen 5 5600H + Radeon Vega iGPU（RADV/Vulkan），ReleaseFast，
机器非独占（CPU 行有 ±10% 波动）。所有 GPU 结果都与 CPU 参考逐元素对拍，`max|diff|` 见各表说明。

> 注：本机还有一块 RTX 3050 Laptop（Vulkan 可见），但 WebGPU 的默认 power preference
> **不保证是独显**（本机默认落在 iGPU），所以每个 GPU 结果都会打印实际选中的 adapter；
> `--adapter high-perf` 可切到独显，实测 GEMM 1.2~2.0x、链式优势从 1.34x 提到 **2.69x**
> （reduce 因搬运主导而无差别）——见下面「adapter 消融」与 `docs/zig-gpu-spike.md` §5。

### adapter 消融（ReleaseFast，同机同二进制，仅改 `--adapter`；3 次取代表值）

同一台机器上两块 Vulkan 设备：iGPU = AMD Radeon (RADV RENOIR)，dGPU = NVIDIA RTX 3050 Laptop。
每个 GPU demo 都会打印实际选中的 adapter（`[integrated|discrete, vulkan]`）。

| 负载 | `--adapter auto`（落 iGPU） | `--adapter high-perf`（dGPU） | 倍数 |
|---|---|---|---|
| GEMM 512³ tiled | 171.3 GFLOP/s | 209.5 GFLOP/s | 1.22x |
| GEMM 512³ tiled_batch | 223.4 GFLOP/s | 424.8 GFLOP/s | 1.90x |
| GEMM 1024³ tiled | 194.7 GFLOP/s | 306.6 GFLOP/s | 1.57x |
| GEMM 1024³ tiled_batch | 223.7 GFLOP/s | 445.0 GFLOP/s | 1.99x |
| chain pipeline 64 reps（chained） | 197.4 GFLOP/s | **490.6 GFLOP/s** | 2.49x |
| chain pipeline 64 reps（submit_cut） | 1.34x | **2.69x** | — |
| reduce 16M sum（gpu_batch） | 20.6 GB/s | 19.9 GB/s | 0.97x（搬运主导） |
| spatial 256K 仅查询（稳态） | 2.64 ms | **0.61 ms** | **4.33x**（随机访存） |

结论：**链式（M0）的价值在独显上更大**（每次 submit+readback 往返更贵：iGPU 1.34x vs dGPU 2.69x），
而在共享内存 iGPU 上测出的数字会低估它。所有对拍不变（GEMM `max|diff| = 0`，chain `rel 2.64e-6`）。

### 判据分档（`--precision`，ReleaseFast；判据不参与计时）

| 负载 | class | `exact` | `tolerant`（默认） | `fast` |
|---|---|---|---|---|
| chain pipeline（GEMM→bias→reduce，16 reps） | accumulated | **MISMATCH**（rel 2.64e-6，abs 2.0e2 / sum 7.57e7） | OK | OK |
| reduce sum（f32 累加） | accumulated | OK（该 shape 实测 `|diff| = 0`） | OK | OK |
| reduce max（无舍入） | discrete | OK（0 容差） | OK | OK |
| GEMM 512³ tiled（tolerance 打印值） | accumulated | tol 0 | tol 7.5e-2 | tol 7.5 |
| 同上耗时（GFLOP/s） | — | 202.6 | 202.8 | 203.5 |

两个要点：

1. **exact 档在 f32 累加上"故意"不过**：GEMM→bias→reduce 的最终标量与 CPU 参考差
   `rel 2.64e-6`（纯累加顺序；WGSL §15.7.5 明确允许实现重结合/融合，所以跨厂商 bit 一致
   不可得）。exact 档的价值就在于此：它把"这里对不上是真 bug"与"浮点顺序差"分开，
   索引/计数/max 走 exact 永远绿灯。
2. **档位不影响性能**：判据在计时区之外，三档 gemm 512³ 为 202.8/202.6/203.5 GFLOP/s（噪声级）。
   "游戏要快"该动的是 adapter 与链式（见下），不是判据。

### M0 消融：saxpy 链（16 MiB/buffer，超过 L3；GB/s 计 3 条流：读 x、读 y、写 x）

| chain_len | cpu_simd | per_call | per_submit | chained  | chained/cpu |
| --------- | -------- | -------- | ---------- | -------- | ----------- |
| 1         | 26.0     | 5.1      | 5.7        | 5.7      | 0.22x       |
| 4         | 37.4     | 5.4      | 15.8       | 16.6     | 0.44x       |
| 16        | 36.2     | 5.8      | 27.9       | 28.4     | 0.78x       |
| 64        | 36.3     | 6.0      | **35.0**   | **35.0** | **0.96x**   |

结论（消融）：`per_call`（旧形态：每步上传+回读）带宽恒定在 ~6 GB/s；
`chained` 随链长增长到 ~35 GB/s，相对 per_call 提升 **5.8x**，并逼近单核 SIMD 的
~36 GB/s —— 即"搬运被摊薄"之后，剩下的差距是这台 iGPU 与 CPU 共享内存的带宽上限，
而不是运行时开销。`per_submit` 与 `chained` 几乎重合，说明在本机驱动上提交本身很便宜，
**收益主要来自常驻与只回读一次**。

### M0 消融：异质链 GEMM→bias→reduce（staged = 每段各自提交+回读；chained = 一次提交）

| 规模 | repeats    | staged                 | chained                | 加速                      | chained GFLOP/s |
| ---- | ---------- | ---------------------- | ---------------------- | ------------------------- | --------------- |
| 256³ | 1 / 4 / 16 | 1.28 / 2.22 / 8.30 ms  | 0.95 / 1.27 / 4.51 ms  | 1.34x / 1.75x / **1.84x** | 35 / 105 / 119  |
| 512³ | 1 / 4 / 16 | 2.90 / 8.58 / 30.75 ms | 1.97 / 6.33 / 22.70 ms | 1.47x / 1.36x / **1.35x** | 136 / 170 / 189 |

异质链对拍用相对容差（`rel ≤ 1e-4`，实测 1.9e-6~2.6e-6，仅 f32 累加顺序差异）。
512³ 的加速比小于 256³，因为计算占比上升、回读占比下降——这也说明链式收益与
"每步数据量 / 计算量之比"直接相关。

### S1 空间索引（ReleaseFast，4096 次查询，半径 2.0，64^3 网格）

| 点数 | cpu 暴力 | cpu 网格 | gpu 网格（1 chain：build+query） | gpu 仅查询（稳态） |
|---|---|---|---|---|
| 16K | 55.2 ms | 3.54 ms (15.6x) | 1.55 ms (35.7x) | 0.50 ms (**111x**) |
| 64K | 222.8 ms | 5.40 ms (41.3x) | 2.17 ms (102.5x) | 0.81 ms (**275x**) |
| 256K | 925.5 ms | 13.3 ms (69.6x) | 6.01 ms (153.9x) | 2.58 ms (**359x**) |

暴力每查询要检查全部点（256K 时 262144 次），网格每查询只检查球内候选
（256K 时 33.5 个），所以加速随规模增长；GPU 计数与 CPU 网格计数**逐元素精确相等**
（u32 无浮点容差问题）。build+query 在一条 Chain 里只提交一次、只回读一次。
CLI 还会验证固定容量邻接表（K=16）：对未被截断的查询比较排序后的邻居集合
（64K/4096 场景下 4066/4096 个查询 MATCH，其余因截断跳过排序比较）。

### 内核基线（ReleaseFast，端到端含上传+回读；对拍 `max|diff| = 0`）

| workload       | cpu_simd     | gpu_simple        | gpu_tiled          | gpu_tiled(稳态)    |
| -------------- | ------------ | ----------------- | ------------------ | ------------------ |
| GEMM 512³      | 60.3 GFLOP/s | 75.5 (1.3x)       | 175.3 (2.9x)       | 216.8 (3.6x)       |
| GEMM 1024³     | 42.9         | 29.1 (0.7x)       | 194.0 (4.5x)       | 226.3 (5.3x)       |
| GEMM 2048³     | 19.2         | 21.0 (1.10x)      | 216.2 (11.3x)      | 234.1 (12.2x)¹     |
| reduce 4M sum  | 45.8 GB/s    | 6.1 (端到端)      | —                  | 18.3 (稳态)        |
| reduce 16M sum | 29.7         | 7.0               | —                  | 26.2（**输给 CPU**）|

参考（历史验收，Debug）：`add` 1<<20 时 CPU SIMD 14.7 GB/s、GPU 端到端 4.7 GB/s、
GPU 稳态 16.2 GB/s —— 这正是 M0 runtime 要解决的问题。

**诚实结论**：GEMM 这类高算力密度内核单次就能赢（CPU SIMD 一侧经 comptime 分块优化后
也快了 1.5~2.4x，但差距仍有 3~5x）；reduce/saxpy 这类纯流式 kernel 现在**端到端输给
CPU SIMD**（CPU 侧 SIMD 归约经多累加器优化后在 16 MiB 上 29.7 GB/s，GPU 稳态 26.2 GB/s），
真正价值是**链式**（5.8x over per-call）而不是单步吞吐。

注：CPU 侧两个 comptime 旋钮（GEMM 分块行数、归约累加器）的实测与判定见
`docs/perf-tooling-and-comptime.md`。
¹ `GEMM 2048³ simple` 的批量列被**单 submit 预算**夹到 1 次 dispatch（实测每次 ~816 ms；
再长会触发 iGPU 驱动 ring timeout，见「限制与已知边界」），tiled 未受影响。

---

## 限制与已知边界

- **单个 submit 的 GPU 工作量必须远小于 ~2 s**（本机 iGPU/RADV 实测：1.63 s 通过、
  2.45 s 触发 `ring gfx timeout` → amdgpu 硬恢复"context is lost" → wgpu-native 在
  `wgpuQueueSubmit` 上**直接 abort，无法用 error scope 捕获**）。CLI 的批量基准按**实测**
  的 per-dispatch 时间把迭代数夹到 1.5 s 预算内并打印提示；库层不管这件事（预算归调用方）。

- GPU 内核目前只有 `f32`；非 f32 会走 CPU SIMD 并记录原因。
- dispatch 的 2D 展平上限：两轴各 `maxComputeWorkgroupsPerDimension`（通常 65535），
  limits 闸门在提交前拦截；buffer 大小同样按 `maxStorageBufferBindingSize`/`maxBufferSize` 校验。
- 读回是**阻塞**的：`ProcessEvents` 轮询 + 30s 墙钟超时；超时会取消挂起的 mapping
  （否则下一次提交会触发 wgpu-native 的 fatal "buffer is still mapped"）。
- 无纹理/采样器、indirect dispatch、atomics/scan/sort、f16/u32、多队列、GPU 计时查询。
- 浏览器端目前只跑固定 size 的 add demo；新内核移植到 wasm 需要同时过 `zig build wasm`
  与 ABI 漂移检查。
- `--auto`/`heuristic` 的选择只针对 `add`；GEMM/reduce/chain 是显式入口。

---

## 开发

贡献规则、模块边界、依赖决策、测试门槛与提交规范见 [AGENTS.md](./AGENTS.md)。
架构背景（为什么要这些层、Blender 节点系统迁移需要补什么）见
**`docs/node-system-migration.md`**。

常用命令：

```bash
zig build test --summary all        # 全绿是硬门槛（含 tree-shake 与 consumer 门禁）
zig build wasm                      # 浏览器路径不得回归
tools/check_abi_drift.sh            # 改绑定后必须过
zig build abi-check                 # 严格模式（CI / 升级依赖）
zig build tree-shake                # CPU-only 消费者对象无 wgpu/WGSL/spatial 符号
zig build consumer-check            # 示例消费者（path 依赖 + CPU-only）能构建并运行
```
