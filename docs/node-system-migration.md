# 从 Blender 节点系统到可移植计算库：可行性、差距与依赖策略

> 目标读者：本库（`computeAccel`）的后续维护者。
> 本文回答三件事：
> 1. 把 Blender 的节点系统迁移到任意平台，对底层计算库意味着什么需求；
> 2. 当前 demo 能否承载这些需求、缺哪些轮子；
> 3. 依赖策略——是否继续用 Zig `@Vector` / wgpu-native（desktop）/ emdawnwebgpu（browser），
>    以及有没有现成的 Zig/C 库可以"无缝继承"这些原语。
>
> 前提：本项目按 greenhouse 处理——**允许大面积重构**，鼓励更好的模块化。
> `SPEC.md` 是当前 demo 的 API 契约；本文件描述的是**下一阶段（v2）架构**，
> M0 落地时应同步升级 SPEC.md。

---

## 0. 结论摘要

- **当前 demo 不能直接承载任何一类节点系统**。它是"7 个固定 kernel 的验证集"
  （add/saxpy/gemm/reduce），缺的不是 kernel，而是三层基础设施：
  **运行时（常驻/链式/分配器）→ 原语（scan/sort/atomics/indirect/纹理）→ 图/IR/调度**。
- 四类节点系统的迁移难度排序：**compositor（最易）< texture < shader ≪ geometry（最难）**。
- 依赖结论：
  - **继续用 Zig `@Vector` + `std.simd`**（CPU 侧只做参考/回退/主机工具，实测瓶颈在内存带宽，不在 SIMD 指令集）；
  - **继续用 wgpu-native（desktop）+ emdawnwebgpu（browser）**，这是当前唯一能"一份 ABI + 一份 WGSL 同时喂 native 与浏览器"的组合；
  - **原语层没有可拿来即用的 Zig/C 库**（详见 §5），必须自研薄层；可复用的集中在 CPU 侧的
    编解码/色彩/去噪/噪声等 C/C++ 库，以及 Halide/naga 这类"结构参考或未来非浏览器路径"。
- 建议里程碑：**M0 运行时 → M1 通用 dispatch → M2 Compositor 最小集 → M3 原语补齐 → M4 Geometry 子集**；
  Shader nodes（SVM→WGSL 编译器）单独立项。

---

## 1. Blender 节点系统清单（需求侧）

以 Blender 4.x 为准，节点树类型共四类；Simulation/Repeat/For-Each 是 Geometry 树内部的
"zone"结构，不是独立树类型：

| 节点树 | UI | 语义本质 | 计算特征 | 备注 |
|---|---|---|---|---|
| **Shader nodes** | Material / World / Light | 逐着色点求值（BSDF 闭包、程序纹理、向量/属性） | 不是"算 buffer"，是编译出一段着色程序；需要导数（`dpdx`）、纹理采样、分支/特化 | 相当于一门 DSL；Blender 自己也要为 EEVEE 写 GPU material compiler |
| **Geometry nodes** | Geometry Nodes | 域（point/face/corner/spline/instance/volume）上的属性场求值与拓扑变更 | 变长数组 + 拓扑 + 动态数量 + 求解器迭代；需要 compaction/scan/sort/atomics/indirect/BVH | 含 Simulation Zone（状态 ping-pong）、Repeat Zone、For-Each Zone；四类里最重 |
| **Compositor nodes** | Compositor | 2D 图像运算（色彩、滤镜、遮罩、跟踪数据） | 逐像素/可分离卷积/金字塔/直方图；天然适合 kernel 融合与分块 | 4.x 已有 GPU 合成后端；语义最规整，最适合做第一个迁移对象 |
| **Texture nodes** | Texture Editor | 程序纹理逐纹素生成（噪声/图案/混合） | 2D/3D dispatch + 确定性噪声/RNG | 2.8 起基本是 legacy，被 shader/geometry nodes 取代；但语义简单、适合练手 |

补充事实与核对方式：

- 四类树在源码里的枚举是 `NTREE_SHADER / NTREE_TEXTURE / NTREE_COMPOSITOR / NTREE_GEOMETRY`，
  建议对照 `source/blender/blenkernel/BKE_node.hh`（`bNodeTreeType`）与 `DNA_node_types.h` 核对；
- Freestyle 的 Line Style 是**修饰器列表**，不是节点树；
- 嵌套 Node Group / Asset 树在迁移时必须展开；
- "任意平台"的现实边界：desktop（wgpu-native 覆盖 Vulkan/Metal/DX12）+ browser（emdawnwebgpu）；
  主机侧移动端可复用 wgpu-native；主机（console）不在覆盖范围。**WebGPU 是最小公分母**，
  IR 必须以它的能力集为基线设计（见 §7）。

---

## 2. 各系统的计算需求 → 原语清单

| 原语 | Compositor | Texture | Shader | Geometry |
|---|---|---|---|---|
| 逐元素算术 / 向量化 | ● | ● | ● | ● |
| 2D/3D dispatch + 边界处理 | ● | ● | ○ | ● |
| texture / sampler / mip / wrap | ● | ○ | ● | ○ |
| 可分离卷积 / 金字塔 / tile 流式 | ● | ○ | ○ | ○ |
| 归约（min/max/sum/直方图） | ● | ○ | ○ | ● |
| scan / sort / compaction | ○ | ○ | ○ | ● |
| atomics + indirect dispatch（动态数量） | ○ | ○ | ○ | ● |
| 邻域查询 / BVH / 空间哈希 | ○ | ○ | ○ | ● |
| 状态 ping-pong（迭代/模拟） | ○ | ○ | ○ | ● |
| 确定性 RNG / 噪声 | ○ | ● | ● | ● |
| 跨后端一致（容差或位精确） | ● | ● | ● | ● |
| 显存常驻 + 多 pass 链式提交 | ● | ● | ● | ● |
| CPU fallback / 混合调度 | ● | ● | ● | ● |

（● 直接需要；○ 次要或不需要。Shader 的"逐元素"其实是编译期生成，不走通用原语。）

横切需求（四个系统都逃不掉）：

1. **类型化属性**：f32/f16/u32/i32、vec2/3/4、SoA/AoS、stride——Blender 的属性系统就是这套；
2. **动态数量**：几何/合成都有"长度在运行时才知道"的域（点数、tile 数、pair 数）；
3. **确定性**：同一节点图换平台/换后端结果要可预期（见 §7.1）；
4. **显存预算**：大图/大几何必须能分块、淘汰、流式，不能"全量常驻"。

---

## 3. 当前库的能力矩阵（对照代码核实）

| 维度 | 现状 | 代码证据 |
|---|---|---|
| 用户持有的常驻 buffer | ❌ `DeviceBuffer` 是 stub，`toDevice/toHost` no-op，`gpu_handle` 未使用 | `src/buffer.zig` |
| GPU buffer 归属 | ❌ 藏在各 kernel 的 `(device, shape)` cache 里，外部拿不到句柄；多 shape 会互相 thrash | `gemm.zig` / `reduce.zig` / `pipeline.zig` 的 `Cache` |
| 多 kernel 链式提交 | ❌ 每次调用一次完整往返；`*Batched` 只是同一 kernel 重复 N 次 | README「每次 GPU 运算的数据路径」 |
| 动态数量 / indirect dispatch | ❌ 绑定里没有 `DispatchWorkgroupsIndirect`，无动态 grid | `src/gpu/webgpu.zig` |
| atomics | ❌ 未使用（WGSL 侧可用 u32/i32 atomic，但需要新的 kernel 模式） | `src/gpu/shaders/*` |
| scan / sort / histogram | ❌ 只有 reduce（两趟，和 scan 同构但未抽象） | `src/gpu/reduce.zig` |
| 纹理 / 采样器 / 格式 | ❌ 绑定只有 storage/uniform buffer；`WGPUTextureFormat` 等只是类型占位 | `src/gpu/webgpu.zig` |
| 数据类型 | ⚠️ 只有 f32 storage + 16B uniform params；无 u32 索引约定、无 f16/bf16、无 vec 打包 | `Params` extern structs / WGSL |
| 分配器 / 资源池 | ❌ 无；一 kernel 一份 shape cache | 同上 |
| 异步 / 重叠 | ❌ readback 阻塞自旋 pump（30s 墙钟超时） | `GpuContext.waitFor/readBuffer` |
| 计时 / 诊断 | ⚠️ 无 GPU timestamp；错误只回 `GpuError`（wgpu message 被丢） | `context.zig` 的 error scope |
| 能力探测 / 回退 | ✅ probe + limits 闸门 + 诚实回退（可提炼为"能力矩阵"） | `context.zig` / `backend.zig` / `bench.zig` |
| 一绑定两目标 + ABI 护栏 | ✅ 31 符号子集 + drift check（**核心资产**） | `tools/check_abi_drift.sh` |
| 正确性方法论 | ✅ CPU 参考 + `max|diff|` + 明确容差 | 各 kernel 的测试与 CLI |
| 已验证 GPU 模式 | ✅ 2D dispatch 展平、workgroup 内存、两趟归约、tiled GEMM | `handoff.md` / README 实测 |

**判定：当前库 = 第 0 层地基的一部分，不能承载节点系统。**

---

## 4. 缺失的轮子（分层清单）

### L0 运行时（数据与内存）
- 用户持有的 `Buffer` + 类型化 `View`（dtype/rank/stride）与生命周期托管；
- 分配器/池、别名（liveness 分析）、显存预算与 LRU/流式；
- 上传策略：小更新（每帧 bone matrices）用 `writeBuffer` 子区间，大更新走 staging + copy；
- 读回策略：按需 readback + 异步提交（不要每次阻塞）。

### L1 执行与调度
- `KernelSpec`（WGSL + entry + bindings + grid 计算）与通用 dispatch，替代每 kernel 手写 encode；
- 多 pass 图：显式依赖、一次提交、ping-pong、最后一个 pass 才读回；
- **indirect dispatch**（动态工作量）与多队列/重叠（native 可选）；
- 绑定槽打包：WebGPU 默认每 stage 只有 **8 个 storage buffer**，融合后的图会先撞这个墙；
- 参数传递：无 push constants，只能 16B uniform → 小参数也要占 buffer，需要 packing 约定。

### L2 原语
- scan/prefix-sum、radix sort、histogram、compaction、atomics（f32 需要 CAS 循环或排序后归约）；
- gather/scatter（WGSL 无 gather 指令，靠索引访存）、稀疏支持；
- 子组原语（subgroup）：WebGPU 支持度有限，**不能当基线**。

### L3 领域（按节点系统）
- 图像：texture/sampler/format/mip、tile 流式、可分离卷积、金字塔、色彩管理；
- 几何：BVH 构建/遍历、空间哈希、邻域查询、属性传播、拓扑邻接、细分/布尔/散射；
- 采样：确定性哈希噪声、蓝噪声、Sobol（跨后端一致是硬要求）。

### L4 图 / IR / 调度（真正的"通用计算范式"）
- 节点图 → IR：类型与域推导、静态分析（活性/依赖/融合机会）、内存规划；
- IR → 各后端：CPU（Zig）/ GPU（WGSL）；未来非浏览器路径（SPIR-V/MSL/HLSL）留接口；
- 特化与分支合并：节点图高度动态（枚举/开关/索引切换），需要常量特化 + 分支化简；
- 缓存/烘焙：中间结果复用（几何节点的 bake、合成的中间缓存）；
- CPU fallback 注册表 + 混合调度（未实现节点走 CPU，结果回灌）。

### L5 正确性 / 确定性
- 容差模式（exact / tolerant / fast）与每算子例外表；
- 跨后端一致性测试基建（同一 IR 跑 CPU/GPU 对拍，已经有的方法论直接扩展）；
- 诊断透传（把 wgpu 的 validation message 带出来）。

---

## 5. 依赖策略（本文件的核心问题）

### 5.1 CPU SIMD：继续用 Zig `@Vector` / `std.simd`

**结论：继续用，且保持薄封装。**

理由与实测依据：

- 我们的 CPU 侧定位是**参考实现 / 回退路径 / 主机工具**，不是"最后的 5%"；
- 实测所有 CPU kernel 都贴在内存带宽上：add 4/8/16 宽 = 23.0/25.4/26.8 GB/s、
  saxpy mul+add/FMA = 25.0/25.4、memcpy 22.9、memset 26.6 GB/s（16 MiB×3 流）；
  4 宽→16 宽只有 +17%，FMA ≈ +1%。**瓶颈是带宽，不是指令集**；
- `@Vector` 跨目标（x86 AVX2/AVX-512、ARM NEON、**wasm simd128**）由 LLVM 处理，
  这对"任意平台"（含浏览器 CPU 回退）是关键；换成 `immintrin.h` intrinsics 会失去它；
- 社区批评（Reddit r/Zig 的 “An introduction to SIMD with Zig” 讨论串：无自动广播、
  无 rcp/gather/bf16/mask、inline asm 不能传向量寄存器）
  对我们当前算子（`+ * max` + 规则访存）**全部不适用**。

工程约定：

- 把 SIMD 收进 `backends/cpu/simd.zig` 薄层（width = `std.simd.suggestVectorLength`、
  `@mulAdd`、`@reduce`），kernel 只调用该层，不直接写 `@Vector`；
- 只有当出现 **transcendental（sin/cos/pow）、rsqrt/rcp、bf16** 等场景，且 profiling 证明有必要时，
  才在该层内部局部引入 `@cImport` C intrinsics，并**保留标量参考实现**用于对拍；
- CPU 侧大 GEMM 若成为瓶颈，再考虑链接 OpenBLAS/MKL（C ABI），不作为默认依赖。

### 5.2 desktop GPU：继续用 wgpu-native

**结论：继续用。**

- 提供稳定的 **C ABI（`webgpu.h`）**，我们只需要 31 个符号的子集，且已有**逐字 ABI 漂移检查**；
- 预编译 `libwgpu_native.so` + pin 版本，构建成本低；底层覆盖 Vulkan/Metal/DX12（平台面足够宽）；
- 备选对比：
  - **Dawn（native）**：与浏览器端（emdawnwebgpu）同源，行为最一致，但要自己构建/获取预编译产物，
    体积与构建复杂度高；可作为未来的"行为对齐"验证后端，不建议现在换；
  - **直连 Vulkan/Metal/DX12**：需要自己维护资源管理 + shader 翻译（naga/SPIRV-Cross/Tint），
    且浏览器那份还是要 WGSL——等于多维护一套，收益只有"少一层 ABI"；
  - **Zig → SPIR-V**（Zig 自带 SPIR-V 后端）：native-only，实验性，且本项目硬性禁用 SPIR-V
    （浏览器拒收），会破坏"一绑定两目标"这个核心资产。
- 可做的优化：把 wgpu-native 改成**运行期 dlopen**（构建解耦、可选后端），并在绑定层预留
  texture / indirect 等符号的扩展位（扩展时必须同步扩 ABI 检查清单）。

### 5.3 browser GPU：继续用 emdawnwebgpu

**结论：继续用（现实上也是唯一选择）。**

- 它是 Dawn 的 emscripten port，是浏览器端唯一"能通过 emcc 链接、C ABI 与 native 同形"的实现；
- 约束要写进 IR 设计：**WGSL only**（拒 SPIR-V）、无 ASYNCIFY → 不能 WaitAny/阻塞，
  只能 `ProcessEvents` + rAF pump、内存/绑定数受限；
- 与 native 的 ABI 一致性是核心资产：**任何绑定扩展（texture/indirect/新 symbol）都必须同时过
  两份头文件的 drift 检查**，否则会出现"只在浏览器崩"的错位。

### 5.4 现成 Zig/C 库能否"无缝继承"原语？

**结论：跨 desktop+browser 的 compute 原语没有现成库；可复用的集中在 CPU 侧 C/C++ 库。**

| 需求 | 候选 | 判定 |
|---|---|---|
| GPU compute 运行时（buffer/pipeline/graph） | Zig 无；Rust `wgpu`/`encase`/`cubecl`；C++ Dawn | **自研薄层**（本库已在做），结构借鉴 wgpu/Dawn 的资源模型 |
| GPU 原语（scan/sort/histogram/compaction） | CUB / rocPRIM（CUDA/HIP，不可移植）、Boost.Compute（OpenCL）、wgpu 生态无 C ABI | **自研**（算法可借鉴），WebGPU 上无现成库 |
| 大 GEMM（CPU） | OpenBLAS / MKL / 自研分块 SIMD | 自研保留；需要时可选链接 BLAS |
| 图像处理（CPU/参考/IO） | `zignal`（Zig, SIMD 图像）、`zigimg`（编解码）、OpenImageIO、OpenCV | 可用于 **CPU 参考与 IO**；GPU 侧自研 WGSL |
| 色彩管理 | OpenColorIO（CPU + GPU 着色器生成；WGSL 支持需按版本核对） | 接入 CPU 侧做参考；GPU 侧按其 LUT/着色器适配 |
| 去噪 / ML 推理 | OIDN、ONNX Runtime、ncnn | 作为外部后端（CPU/可选 GPU），不进 WebGPU 核心 |
| 噪声 / RNG | FastNoise2（CPU SIMD）；GPU 自研哈希 | CPU 参考可用；**GPU 侧必须自研**（确定性关键） |
| 几何 / BVH | Embree（CPU/SYCL）、OpenVDB、libigl、Bullet/Jolt | CPU 侧可复用；GPU/WebGPU 侧自研 |
| 图调度 / 融合 | Halide（图像 DSL，CPU/CUDA/OpenCL/Metal；**无 WebGPU**）、TVM/MLIR | 不可直接用；**借鉴其 schedule 思想**（fuse/tile/vectorize）设计自研 IR |
| 着色器翻译（未来非浏览器路径） | naga / SPIRV-Cross / Tint | 只有放弃浏览器时才值得引入 |
| 图形/游戏框架 | `zig-gamedev`（zgpu/zmath/zmesh，Dawn 系）、`mach` | 图形向、依赖重、版本churn 大；不解决 compute 原语，不建议引入 |

直白说：**"原语层"是这个项目的自研职责，也是它存在的意义**；能外包的只有
CPU 参考实现、编解码、色彩、去噪这些"领域 C/C++ 库"，以及未来的 shader 翻译。

### 5.5 依赖策略汇总

| 依赖 | 决策 | 理由 |
|---|---|---|
| Zig `@Vector` + `std.simd` | **保留**（薄封装） | 零依赖、跨目标、wasm SIMD、内存带宽受限 |
| wgpu-native（desktop） | **保留** | C ABI 小、预编译、覆盖 Vulkan/Metal/DX12、有 drift check |
| emdawnwebgpu（browser） | **保留** | 唯一现实的"浏览器 + 同形 C ABI";约束写进 IR 基线 |
| 原语/运行时（scan/sort/BVH/texture…） | **自研** | 无可移植的现成库 |
| CPU 领域库（OIIO/OCIO/OIDN/FastNoise2/zignal/zigimg） | **按需接入** | 强在 CPU 侧，接口是 C ABI |
| Halide / naga / Dawn(native) / SPIR-V | **攒着** | 只有放弃浏览器或要"行为对齐"时才引入 |

---

## 6. 建议架构（greenhouse：允许大重构）

### 6.1 目标模块布局

```
src/
  root.zig                  # 窄公共 API（对外只暴露 runtime + primitives）
  capability.zig            # 能力探测/limits/feature 矩阵（现 context.zig 的 probe 部分）
  runtime/
    device.zig              # Device：instance/adapter/queue/limits/feature + 生命周期
    buffer.zig              # Buffer / View(T)/Allocation / pool / staging
    graph.zig               # 多 pass 记录、依赖、提交、readback 策略
    dispatch.zig            # KernelSpec + grid 计算（1D/2D/indirect）
  backends/
    cpu/simd.zig            # @Vector 薄层（width/FMA/reduce）
    cpu/kernels/…           # CPU 实现（参考 + 回退）
    webgpu/abi.zig          # 现 webgpu.zig（可扩展子集 + drift check）
    webgpu/device.zig       # 资源与生命周期
    webgpu/pipeline.zig     # KernelSpec → pipeline/bind group 缓存
  primitives/
    elementwise.zig gemm.zig reduce.zig scan.zig sort.zig
    image/{tile,conv,pyramid}.zig      # 领域无关
    spatial/{bvh,hash,proximity}.zig   # 按需（S1）：只在有具体消费方时新增
  determinism.zig           # exact/tolerant/fast + 每算子容差表
  selection.zig             # 现 backend.zig + bench.zig（保留为回归/自检工具）
docs/ tools/                # 保持
```

### 6.2 与现有代码的映射

| 现有 | 去向 |
|---|---|
| `gpu/context.zig` 的 probe/limits/回退 | `capability.zig` + `runtime/device.zig` |
| `gpu/context.zig` 的 create/submit/readBuffer 辅助 | `backends/webgpu/device.zig`（保留，已经是共享 plumbing） |
| `gpu/pipeline.zig`（add/saxpy） | `backends/webgpu/pipeline.zig` + `primitives/elementwise.zig` |
| `gpu/gemm.zig` / `gpu/reduce.zig` | `primitives/gemm.zig` / `primitives/reduce.zig`；**去掉 shape-keyed cache**，改由 runtime 分配 |
| `engine.zig` | `backends/cpu/kernels/` + primitives 的 CPU 实现 |
| `backend.zig` + `bench.zig` | `selection.zig`（demo 的自动选择保留为工具） |
| `buffer.zig`（stub） | 被 `runtime/buffer.zig` 取代 |
| `tools/check_abi_drift.sh` | 保留并成为**任何绑定扩展的强制门禁** |

### 6.3 里程碑

| 里程碑 | 归属 | 内容 | 验收 |
|---|---|---|---|
| **M0 运行时**（原 Step 1/T8） | 库内 ✅ | 用户持有的常驻 Buffer、`toDevice/toHost`、多 kernel 一条 command buffer、一次回读、资源池雏形 | 3 段链（如 gemm→bias→reduce）与 CPU 参考对拍；链长增加时 GPU 端到端斜率变好 |
| **M1 通用 dispatch** | 库内 | `KernelSpec` + 通用 grid/绑定 + 异步 readback + GPU timestamp | 同一 kernel 描述在 CPU/GPU 都能跑且对拍通过；加新 kernel 只写 1 WGSL + 1 spec |
| **M2 图像与图调度** | 库内 | 2D tile、逐像素/可分离卷积/金字塔、kernel 融合、通用 op-graph 调度（IR 雏形） | 一张融合后的图像 op-graph 与逐算子实现逐元素对拍；融合前后 profile |
| **M2' Blender 合成节点** | 中间件 | 合成节点语义、OCIO 色彩策略、图像源/缓存 | 与 Blender 参考输出对拍；不进本库里程碑 |
| **M3 原语补齐**（进行中） | 库内 | ✅ scan（u32 排他前缀和，3 段链 + CPU 对拍）、✅ compaction（flags→scan→scatter+total，count 可作 indirect 参数）、✅ indirect dispatch（ABI 32 符号 + 测试）、✅ atomics（网格计数/散射已验证）；⬜ sort、texture/sampler/format/mip | 纹理噪声跨后端一致；sort 与 CPU 对拍 |
| **S1 空间原语（按需）**（进行中） | 库内 | ✅ 均匀网格索引（GPU：clear → atomics count → scan → atomic scatter → 半径查询；CPU 参考对拍；消融 111x→359x vs 暴力）；⬜ BVH 构建/遍历、空间哈希、kNN | 该消费方的 workload 对拍 + 消融（相对暴力解法） |
| — | 中间件 | Blender 域模型（point/face/corner/spline/instance/volume）、属性传播语义、字段求值、Simulation/Repeat/For-Each zone、bake/cache 策略、节点解析与节点语义 | 不在本库；由消费方实现与验收 |
| 独立立项 | 库外 | **Shader nodes**：SVM→WGSL 编译器 + 纹理/采样/导数 | 不在本库主线内，单独立项评估 |

**范围边界（判断一个东西该不该进本库）**：

1. 本库负责**与领域无关的并行计算**：设备运行时（常驻/链式/limits/回退）、
   并行原语（元素级/GEMM/归约/scan/sort/纹理）、通用 op-graph 调度。
   它不知道"节点""域""属性"是什么。
2. 进库的门槛：**(a)** 语义不引用具体消费方（Blender/物理/渲染/碰撞）；
   **(b)** 至少有一个明确的 workload 需要它（按需驱动，不预埋）。
   BVH/空间哈希满足 (a)，但必须等 (b) 出现才做。
3. 所有 Blender 语义（节点类型、域模型、属性传播、色彩策略、外部库集成）
   属于**中间件**；本库只提供它们需要的原语与调度。`Shader nodes` 是独立的
   编译器项目，不在本库范围。

**Step 5b（M1 迁移，已实现）**：`gpu/gemm.zig` 迁到 runtime：新增 `Kernels`
（编译好的 simple/tiled pipeline + `bind`，可在任意 Chain 里用调用方自己的 buffer
组合）、`gridFor`（2D 展平 / tile grid）、`VariantCache`（按 device+shape 的常驻 buffer
与两个 bind group），slice API 行为不变。实测无回退：512³ tiled 170.8 GFLOP/s e2e /
223.2 稳态，1024³ 189.8 / 225.3（ReleaseFast，max|diff|=0）。

**Step 5a（M1 迁移，已实现）**：`add`/`saxpy` 从 `gpu/pipeline.zig` 的内建
kernel cache（`GpuContext.pipelines/resources` + `layoutEntries/ensurePipeline/
ensureResources/dispatchMany`）迁到 runtime：新 `primitives/elementwise.zig` 只有
"每 kernel 一个 Runner（Kernel + 常驻 buffer + bind group，按 device/size 缓存）+
execute"，`gpu/pipeline.zig` 退化为兼容 shim；`GpuContext` 删掉 pipelines/resources
双轨（-34 行）。行为不变（每次调用上传/回读），实测 add 端到端 3.85 GB/s、
gpu_batch 22.8 GB/s（ReleaseFast，1<<20）。gemm/reduce 待迁。

**Step 4（M3 compaction，已实现）**：`src/primitives/compaction.zig` 用 scan +
scatter 实现 stream compaction，并把压缩后的长度写进一个 `Indirect` usage 的
3×u32 count buffer（`[len, 1, 1]`），可直接喂给 `Chain.dispatchIndirect`；
4 字节字搬运，u32/f32 载荷位精确保持。测试：5000 元素对拍 + "无 flag 时长度为 0"。

**Step 3（S1 均匀网格，已实现）**：`src/spatial/grid_hash_gpu.zig` 把 CPU 参考的
数据布局搬上 GPU：5 个 kernel（clear / count(atomicAdd) / scan / copy / scatter(atomicAdd)）
+ 查询 kernel，全部记录进一条 Chain；AoS `vec4<f32>` 布局让查询 kernel 保持在
WebGPU 的 8 storage buffer 基线内。实测（ReleaseFast，4096 queries，r=2，64³ cells，
与 CPU 网格计数逐元素精确一致）：16K/64K/256K 点时 GPU 端到端 35.7x/102.5x/153.9x
vs 单线程暴力，仅查询稳态 111x/275x/359x。BVH/空间哈希暂缓（等具体消费方）。

**Step 2（M3 起步，已实现）**：`src/primitives/scan.zig` 提供 u32 排他前缀和
（block_scan → block_scan_global → scan_apply 三段记录进一条 Chain，CPU 参考对拍，
含 1~4097 的边界尺寸）；`Chain.dispatchIndirect` + `wgpuComputePassEncoderDispatchWorkgroupsIndirect`
绑定（ABI 子集 31 → 32 符号）。踩到并记录了 wgpu 的 usage-scope 排他规则：
indirect buffer 不能在同一 dispatch 里再绑成 storage（拆两次 dispatch/两个 shader）。

**Step 1（模块化，已实现）**：发布形态拆成一个包两个 module——
`computeAccel`（CPU/GPU/runtime/原语）与 `computeAccel_spatial`（S1 空间原语），
配套 `zig build tree-shake` 门禁（CPU-only 消费者对象 10,792 B、0 个 wgpu/WGSL 符号；
含 GPU 引用时 201,872 B、42 个 wgpu 符号）。后续新领域一律新增 module，
主 module 不反向依赖。当前 `computeAccel_spatial` 已有 CPU 参考实现
（均匀网格 build + 半径查询），GPU 版本（atomics + scan + scatter）是下一步。

**M0 落地状态（已实现，见 README 实测）**：实际代码是 `src/runtime.zig` +
`src/runtime/{buffer,kernel,chain}.zig`（不重命名现有 `gpu/context.zig`，用
`runtime.Device` 别名指向它），加 `src/chain_bench.zig` 与 `--kernel chain` 消融；
验收用 saxpy 链（per_call / per_submit / chained × 链长 1/4/16/64）和
GEMM→bias→reduce 异质链，结果与 CPU 参考对拍。`backends/`、`primitives/` 的目录级
重组留到 M1/M2，避免一次性大搬家掩盖行为变化。

---

## 7. 风险与验证点

### 7.1 跨后端确定性（最高风险）

- WGSL 的 `sin/cos/pow/exp` 由驱动实现，不同 GPU 结果可差几个 ulp；
  FMA 收缩、归约求和顺序、纹理过滤、denormal 处理都会造成差异；
- 对策：定义三档模式 `exact`（同后端可复现）/**`tolerant`（默认，逐算子容差表）**/`fast`；
  对"会改变控制流"的计算（散射采样、哈希、排序键）强制用整数/哈希运算，保证跨后端一致；
- 测试基建：把现有"CPU 参考 + `max|diff|`"扩成"同一 IR 跑所有后端对拍"。

### 7.2 外部库与语义覆盖

- Blender 节点背后有一批 C++ 库（OCIO、OIDN、OpenVDB、Bullet/Jolt、图像格式）；
  迁移必须定义**GPU-native 子集 + CPU fallback**，不能假设全都能上 GPU；
- 混合调度（GPU 结果回灌 CPU、CPU 结果回灌 GPU）要有明确的数据路径与预算。

### 7.3 WebGPU 最小公分母

- 每 stage 8 个 storage buffer、无 push constants（16B uniform）、f16 需 feature、
  无 f64、subgroup 支持有限、无 GPU 内 malloc、无动态并行（只能 indirect）；
- IR 必须以这些约束为设计基线，native 只当"超集优化"。

### 7.4 浏览器端

- 无 ASYNCIFY → 不能 WaitAny/阻塞，只能 `ProcessEvents` + rAF pump（现有 30s 墙钟超时模式可复用）；
- 大图/大几何的内存上限更紧，必须支持分块/流式；
- wasm 端目前只跑固定 add demo，任何新能力都要重跑 `zig build wasm` 并保持 ABI 一致。

### 7.5 许可

- Blender 是 GPL-2.0-or-later：**不要复制 Blender 源码**；本库（MIT 风格）可以独立实现语义，
  但"Blender 业务层胶水"（中间件）与其链接方式、以及是否分发，需要法务确认。

---

## 8. 待验证清单（写代码前先落实）

- [ ] 对照 `BKE_node.hh` 核对四类节点树与 zone 结构（含 Freestyle 的排除理由）；
- [ ] 各候选 C/C++ 库的**实际版本、许可、是否有预编译**（OIIO/OCIO/OIDN/FastNoise2/Embree/zignal/zigimg）；
- [ ] OCIO / OIDN 的 WGSL/WebGPU 支持现状（按版本实测）；
- [ ] WebGPU 在目标浏览器/驱动上的 feature 可用性（f16、indirect、subgroup、绑定数上限）；
- [ ] 一次真实节点图的样本（导出 Blender 的节点树 JSON + 参考输出图），作为 M2 的对拍基准；
- [ ] 决定 IR 的宿主形态（Zig 内存结构 + 是否序列化）与图编辑入口（谁生成 IR：Blender 插件导出？）。
