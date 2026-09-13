# computeAccel — 可移植计算内核与常驻链式运行时（Zig + WebGPU）

一个 Zig 0.16 库/演示：CPU SIMD 与 WebGPU 计算共用一套内核定义，
**同一份绑定的 C ABI 与同一份 WGSL 同时编译到 native（wgpu-native）与浏览器（emdawnwebgpu）**。
除了逐元素/GEMM/归约内核，还提供 M0 运行时：**常驻显存 buffer + 多 dispatch 一次提交、
一次回读**，用于把"每步一次搬运"的工作负载变成"搬一次算很多步"。

- 使用/安装：本文档（面向调用者）
- 开发与贡献规则：`AGENTS.md`
- 架构与后续路线（Blender 节点系统迁移评估）：`docs/node-system-migration.md`
- 依赖说明：`deps/README.md`

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
再用 Chrome 打开 `http://127.0.0.1:8080/shell.html`（页面会跑 CPU SIMD 与 WGSL add 并对拍，
显示 `GPU add: MATCH` 即通过）。

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
```

`--kernel chain --chain pipeline` 跑的是 **GEMM → bias → reduce** 四段异质链
（三种不同内核、两种输出形状），用于验证多内核依赖顺序与"一次提交"。

---

## 实测

环境：AMD Ryzen 5 5600H + Radeon Vega iGPU（RADV/Vulkan），ReleaseFast，
机器非独占（CPU 行有 ±10% 波动）。所有 GPU 结果都与 CPU 参考逐元素对拍，`max|diff|` 见各表说明。

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

异质链对拍用相对容差（`rel ≤ 1e-4`，实测 1.9e-6~3.7e-6，仅 f32 累加顺序差异）。
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

### 内核基线（ReleaseFast，端到端含上传+回读；对拍 `max|diff| = 0`）

| workload       | cpu_simd     | gpu_simple   | gpu_tiled     | gpu_tiled(稳态) |
| -------------- | ------------ | ------------ | ------------- | --------------- |
| GEMM 512³      | 33.3 GFLOP/s | 79.0 (2.4x)  | 173.0 (5.2x)  | 218.2 (6.6x)    |
| GEMM 1024³     | 17.8         | 28.8 (1.6x)  | 196.3 (11.0x) | 226.8 (12.7x)   |
| GEMM 2048³     | 12.5         | —            | 215.1 (17.3x) | 230.4 (18.5x)   |
| reduce 4M sum  | 30.5 GB/s    | 6.1 (端到端) | —             | 14.3 (稳态)     |
| reduce 16M sum | 25.5         | 7.0          | —             | 25.1（≈打平）   |

参考（历史验收，Debug）：`add` 1<<20 时 CPU SIMD 14.7 GB/s、GPU 端到端 4.7 GB/s、
GPU 稳态 16.2 GB/s —— 这正是 M0 runtime 要解决的问题。

**诚实结论**：GEMM 这类高算力密度内核单次就能赢；reduce/saxpy 这类纯流式 kernel
在共享内存的 iGPU 上只能到"打平 CPU SIMD"，真正价值是**链式**（5.8x over per-call）
而不是单步吞吐。

---

## 限制与已知边界

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
