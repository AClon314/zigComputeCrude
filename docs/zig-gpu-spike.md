# Zig 作为 GPU kernel 语言：可行性 spike（2026-09-19）

背景：读了 [alichraghi / "Zig and GPUs"](https://alichraghi.github.io/blog/zig-gpu/)（要点见 §1），
该文宣称 Zig 现在有三条 GPU 代码生成路径（SPIR-V / PTX / AMDGCN），并且 `std.gpu`
提供 shader 侧内置量。**本文回答：`std.gpu` 能不能简化我们的技术路线？**
所有结论都来自本机实测（版本与命令见 §2，可复现）。

---

## 0. 结论摘要

**不能，且不建议迁移。** `std.gpu` 替换的是"kernel 用什么语言写"，**不是** host 绑定、
不是运行时、不是原语——它救不了浏览器端，也换不掉 wgpu-native。四条硬理由：

1. **浏览器只吃 WGSL**：现行 WebGPU 规范里 **完全不含 "SPIR-V"** 字样，`GPUShaderModule`
   的代码只能是 WGSL 文本。Zig 的三条路径产出的是 SPIR-V / PTX / AMDGCN，
   **没有一条能进浏览器** → 采用它等于放弃"一绑定两目标"这个核心资产。
2. **就算想中转（SPIR-V → naga → WGSL）也走不通**：Zig 的 SPIR-V 输出使用
   `OpMemoryModel PhysicalStorageBuffer64` + `PhysicalStorageBufferAddresses` capability，
   而 naga 的 spv-in 支持清单（`SUPPORTED_CAPABILITIES`，35 项）里**没有**这一项；
   WebGPU 也没有 buffer-device-address 概念。
3. **`std.gpu` 没有 host 侧 API**（无 device/queue/pipeline/buffer/mapping/submit/readback）——
   它只是 shader 语言层；我们的 `runtime/{buffer,kernel,chain}.zig` 这一层在 Zig 生态里
   依然无人提供（与 `node-system-migration.md` §5.6 的结论一致）。
4. **工具链未成熟到可押注**：本机 Zig 0.16.0 实测 `std.gpu.executionMode()` **根本发不出**
   `OpExecutionMode`（assembler 直接拒绝，见 §3.1），`@atomicRmw` 是 TODO，
   `addrspace(.shared)` 变量让编译器 panic，PTX 目标触发 `LLVM ERROR`，
   AMDGCN 在**标准 global-index 写法**（`@workGroupId(0) * @workGroupSize(0)`）上 panic。

**但这次 spike 之外带出两个可以立刻做、且收益更大的东西**（都不是 Zig-GPU 相关的）：

- **§5 adapter 选择**：我们的 `wgpuInstanceRequestAdapter(instance, null, ...)` 在双显卡机器上
  永远落在 iGPU。实测（同一二进制、同一 512³ GEMM+chain）：dGPU 链式吞吐 **488.7 vs 196.4 GFLOP/s
  （2.49x）**，而且**链式相对分步的优势从 1.39x 变成 2.69x** —— 我们旗舰论据在 iGPU 上被低估了。
- **§6 确定性契约**：WGSL 规范 §15.7.5 明确**允许重结合与融合**（"An implementation may
  reassociate operations"），且 §17.5.32 的 `fma` 精度是"继承自 x*y+z"（**不保证**是真 FMA）。
  这解释了为什么"浮点跨厂商 bit 级一致"在 WGSL 上不可得，我们的"对拍 + 显式容差"是对的，
  但**整数/索引类路径应当按 bit 精确（== 0）比**，这是个可以立刻收紧的门禁。

---

## 1. 博客要点（作为对照输入）

| 论点 | 我们的核查 |
|---|---|
| SPIR-V 后端（自研，约 4 年）已可用于实验；`spirv64-vulkan-none` / `spirv64-opencl-none` | 能编译，但**产不出合法 compute 模块**（§3.1） |
| 通过 LLVM 直接生成 PTX（`nvptx64-cuda-none`）与 AMDGCN（`amdgcn-amdhsa-none`），运行期加载（`cuModuleLoadData` / `hipModuleLoad`） | PTX **编译即崩**；AMDGCN 能出 HSA ELF，但标准索引写法崩（§3.2） |
| Vulkan baseline 下通过约 50% behavior tests，OpenCL ~75%（作者说不会显著提高） | 与我们实测的"缺 decoration / LocalSize / atomics"一致 |
| 挑战：显式 addrspace；Vulkan 无 `OpPtrCastToGeneric`，**临时把指针都当 local(Function)**；`fma/sqrt/exp/log` 在 Vulkan 不保证正确舍入 | 与我们 §6 的容差契约同源（WGSL 也有同类宽松） |
| Roadmap：composite integers、`spirv-val` 不应失败、**CUDA/HIP runtime 绑定**、**stdlib 里加 prefix sum / reduction / matmul** | 后两条值得关注：前者仍是"绑定"而非运行时；后者的清单**正是我们的 `primitives/`**（见 §4.3） |

---

## 2. 复现方法

环境：Zig **0.16.0**（`~/.pixi/envs/zig`，`x86_64-linux`）；`spirv-val`/`spirv-dis` 来自
`~/.pixi/envs/shaderc/bin`（spirv-tools 2026.3）；机器 = Ryzen 5 5600H + **Radeon Vega iGPU
(RADV)** + **RTX 3050 Laptop (driver 610.57.04)** 双显卡；非独占。

```bash
# SPIR-V（Vulkan）：
zig build-obj -target spirv64-vulkan-none -mcpu vulkan_v1_2+int64 -ofmt=spirv -fno-llvm k.zig
~/.pixi/envs/shaderc/bin/spirv-val --target-env vulkan1.2 k.spv

# PTX（博客的原始命令）：
zig build-lib -dynamic -target nvptx64-cuda-none -mcpu sm_86 -femit-asm -fno-emit-bin -fno-ubsan-rt k.zig
# PTX（换成 build-obj）：
zig build-obj -target nvptx64-cuda-none -mcpu sm_86 -femit-asm -fno-emit-bin -fno-ubsan-rt k.zig

# AMDGCN：
zig build-obj -target amdgcn-amdhsa-none -mcpu gfx90c k.zig
```

内核源码用两种形态：博客原样的 `export fn k() callconv(.kernel) void {}`，以及
`std.gpu` 内置量/addrspace 版本（如 `pub const x: *addrspace(.storage_buffer) f32 =
@extern(...)`、`std.gpu.global_invocation_id`、`@workGroupId/@workGroupSize/@workItemId`）。

---

## 3. 实测结果

### 3.1 SPIR-V：能编出来，但发不出合法 compute 模块

| 检查项 | 结果 |
|---|---|
| `-target spirv64-vulkan-none -ofmt=spirv -fno-llvm` 空 kernel | ✅ 产出 `.spv` |
| 用 `addrspace(.storage_buffer)` + `global_invocation_id` | ✅ 能编（但见下） |
| 内存模型 | ❌ `OpMemoryModel PhysicalStorageBuffer64 GLSL450` + `OpCapability PhysicalStorageBufferAddresses` |
| `spirv-val --target-env vulkan1.2` | ❌ `GLCompute ... requires LocalSize` |
| 手工补 `OpExecutionMode <ep> LocalSize 64 1 1` 后 | ❌ `StorageBuffer OpVariable has illegal type`（`.storage_buffer` 变量必须是 `OpTypeStruct` 且带 `Block` 等 decoration） |
| 换成 `extern struct` 包一层再打补丁 | ❌ `In Logical addressing, variables can only allocate a pointer to the StorageBuffer or Workgroup storage classes`（编译期把 `Target.Os.*` 之类的 Function 变量也带进了模块） |
| 绑定 decoration（`DescriptorSet`/`Binding`） | ❌ 0.16.0 的 `std.gpu` **没有** `binding()`/`location()`（master 才补上）；`@extern` 的 `ExternOptions` **没有** binding 字段；`asm volatile ("OpDecorate ...")` 是唯一途径 |
| `std.gpu.executionMode(entry, .{.local_size=...})` | ❌ **该 API 不可用**：它用 `asm volatile` 实现，而 SPIR-V assembler 对 `OpExecutionMode` 直接 `fail("cannot set execution mode in assembly")` |
| `@atomicRmw`（我们 `grid_count/grid_scatter` 的 `atomicAdd`） | ❌ `error: TODO (SPIR-V): implement AIR tag atomic_rmw` |
| `var x: T addrspace(.shared)`（我们 `scan/reduce/gemm_tiled` 的 workgroup 内存） | ❌ **编译器 panic**（`access of union field 'extern' while field 'undef' is active`），`Cannot print stack trace` |
| in-tree 用例 | 全仓库 `rg 'executionMode\(|gpu\.binding'` → 除 `lib/std/gpu.zig` 自身外**零引用**（无测试、无示例） |

> 即：我们 11 个 WGSL kernel 里，用 workgroup 内存 + barrier 的 3 个（scan / reduce / gemm_tiled）
> 和用 atomic 的 2 个（grid_count / grid_scatter）**在 Zig SPIR-V 路径上都写不出来**；
> 其余的最多只能编出一个**非法**模块。

### 3.2 PTX / AMDGCN

| 路径 | 命令 | 结果 |
|---|---|---|
| PTX | 博客原样 `build-lib -dynamic` | ❌ `error: dynamic linking unavailable on the specified target` |
| PTX | 改 `build-obj`（空 kernel） | ❌ `LLVM ERROR: NVPTX aliasee must be a non-kernel function definition`（**进程 core dump**） |
| AMDGCN | 空 kernel，`-mcpu gfx90c` | ✅ 产出合法 HSA ELF（`OS/ABI: AMD HSA`，带 sramecc warning） |
| AMDGCN | `x[0] = 1.0`（指针参数 + 存储） | ✅ |
| AMDGCN | `x[0] = a*y[0] + x[0]`（浮点乘加） | ✅ |
| AMDGCN | `_ = @workItemId(0)` / `_ = @workGroupId(0)` / `_ = @workGroupSize(0)` / 两者相加 | ✅ |
| AMDGCN | `_ = @workGroupId(0) * @workGroupSize(0)`（**标准 global-index 写法**） | ❌ **编译器 panic**（`reached unreachable code`，core dump） |

### 3.3 为什么"中转成 WGSL"也不行（把这条路彻底封死）

- naga spv-in 的 `SUPPORTED_CAPABILITIES`（35 项，含 `Int64`/`Float16`/`GroupNonUniform*`/
  `RuntimeDescriptorArray` 等）**不包含 `PhysicalStorageBufferAddresses`**；Zig 的输出正好用它。
- WebGPU 规范全文没有 SPIR-V（本文按 `w3.org/TR/webgpu/` 现行版本核对）；
  `GPUShaderModule` 的代码是 WGSL 文本，`wgslLanguageFeatures` 也只描述 WGSL 扩展。
- 因此"Zig kernel → SPIR-V → naga → WGSL → 浏览器"这条假想链路，**在第一步的地址模型就断了**。

---

## 4. 对当前架构的判断

### 4.1 不改（保持 WGSL 单一定义 + 手写 C ABI 子集）

`std.gpu` 带来的**唯一真实好处**是"kernel 不再用字符串写"：类型化 binding、编译期检查、
CPU 参考实现与 GPU kernel 可能共用一份源码（这才是"对拍"的终极形态）。但这些好处需要
①Zig 后端能产出合法且可绑定缓冲区的模块 ②浏览器能接受它的输出 ③workgroup/atomic 可用——
三条今天都不成立，而代价是**放弃核心资产 + 引入随 Zig master 漂移的实验性依赖**。

### 4.2 保留的一个"接缝"

我们的 `runtime.Kernel` 现在接收 `(shader_code, entry, bindings, workgroup_size)`，
**这个签名本身就是前端无关的**。建议在文档里把它明确成扩展点，并写清"新前端准入条件"，
这样将来若条件变化，只需要写一个 front-end，不动 Chain/primitives：

- 必须能产出 **WGSL**（或让浏览器端另有等价路径）；
- 不能用 PhysicalStorageBuffer/buffer device address 一类 WebGPU 没有的概念；
- 必须支持 workgroup 内存 + barrier + 整数 atomic（我们的 scan/reduce/gemm/grid 需要）；
- 必须能显式声明 binding 布局（set/binding → 与 `Kernel.bindings` 顺序一致）。

### 4.3 一个顺带的"上游信号"

博客的 roadmap 写了"**在 stdlib 里加一批能在 GPU 上跑的通用算法（prefix sum、reduction、
matmul…）**"。那正是我们的 `primitives/{scan,compaction,elementwise}` + gemm/reduce。
这既是认可（清单选得对），也是提醒：**若 Zig 官方将来真做了这一层，我们的差异化必须落在
host 运行时（Buffer/Kernel/Chain）、对拍契约、以及 spatial 这类领域无关原语上**，
所以 primitives 应当保持"薄、零依赖、可单独抽出"的形态（现在已经如此）。

---

## 5. 顺带发现（更值得做）：adapter 选择——我们一直在用 iGPU

### 5.1 事实

`src/gpu/context.zig:264` 是 `wgpuInstanceRequestAdapter(self.instance, null, ...)`，
即 **options = NULL（等价 `powerPreference = Undefined`）**。本机双显卡实测
（独立小 C 探针，链接 `vendor/wgpu-native/lib/libwgpu_native.so`，`wgpu-native v29.0.1.1`）：

| `powerPreference` | 选中的 adapter |
|---|---|
| `NULL` / `Undefined` / `LowPower` | RADV RENOIR（AMD Radeon Graphics），`adapterType=2` Integrated |
| `HighPerformance` | **NVIDIA GeForce RTX 3050 Laptop GPU**，`adapterType=1` Discrete |

也就是说 README 里那张"iGPU（RADV/Vulkan）"的表 **不是因为机器只有 iGPU，而是因为默认选择落在 iGPU**。

### 5.2 实测影响（LD_PRELOAD 注入 `HighPerformance`，仓库代码未改；同一 `zig-out/bin/computeAccel`）

`GEMM 512³ tiled, iters=10`：

| adapter | gpu_tiled | gpu_tiled_batch | 对拍 |
|---|---|---|---|
| iGPU（默认） | 148.9 GFLOP/s | 220.7 GFLOP/s | `max|diff| = 0` |
| dGPU（HighPerformance） | 211.3 GFLOP/s (1.42x) | 425.1 GFLOP/s (1.93x) | `max|diff| = 0` |

`GEMM 1024³ tiled, iters=5`：iGPU 167.8 / 217.6 → dGPU **295.4 / 441.9 GFLOP/s**（1.76x / 2.03x）。

`--kernel chain --chain pipeline`（64 repeats，3 次取代表值；方差 ≤0.6%）：

| adapter | staged | chained | submit_cut | chained GFLOP/s | verify |
|---|---|---|---|---|---|
| iGPU | 121.4 ms | 87.5 ms | **1.39x** | 196.4 | OK (rel 3.70e-6) |
| dGPU | 94.5 ms | 35.2 ms | **2.69x** | **488.7** | OK (rel 3.70e-6) |

`--kernel reduce --size 16777216`：iGPU 19.8 GB/s vs dGPU 19.9 GB/s —— **无差别**，
因为这条路径是"1 次上传 + 5 次归约 + 1 次回读"的搬运主导，换 GPU 不改 PCIe 带宽。

### 5.3 结论与建议

1. **链式（M0）的价值被 iGPU 低估**：iGPU 上 1.39x、dGPU 上 2.69x —— 因为 dGPU 的
   每次 submit+readback 往返更贵。README/§4 的旗舰数字应补一行 dGPU 实测。
2. 需要给 GPU 上下文加 **adapter 选择**：`AdapterPreference = { auto, high_performance,
   low_power, vendor_id }`，并把 `description / device / adapterType / vendorID / deviceID`
   透出到 `probe()`（现在只透出 `backend_type`/`vendorID`，看不出选中了哪块卡）。
   注意 `probe()` 是**全局单例缓存**，加偏好后缓存键必须包含它。
3. 不建议硬编码 `HighPerformance`：可移植库的调用方（如节点系统）大概率要与显示/渲染
   用同一块卡；应当"可指定 + 可 benchmark 选择"，并把实测做成消融（`--adapter` 参数）。
4. **方法学注意**：§5.2 用 LD_PRELOAD 注入（不改仓库）、单次/3 次取代表值、机器非独占；
   正式回填 README 时应重测并标注。

---

## 6. 顺带发现（更值得做）：确定性契约按 WGSL 规范收紧

规范原文（§15.7.5 Reassociation and Fusion）：

> Reassociation is the reordering of operations in an expression such that the answer is the
> same if computed exactly… **An implementation may reassociate operations. An implementation
> may fuse operations** if the transformed expression is at least as accurate as the original
> formulation.

以及 `17.5.32 fma` 的精度是 **"Inherited from `x * y + z`"**（规范明确允许实现"先乘后加"，
**不要求真的融合**）。对照我们的现状：

- 我们在 CPU 侧刻意用 `@mulAdd` 固定"融合"语义，但**WGSL 侧无法固定**（写 `fma()` 也一样）；
- 因此"CPU vs GPU 逐元素 bit 一致"对含乘加的 f32 kernel **在规范层面就不可保证**，
  跨厂商（RADV vs NVIDIA vs D3D12/DXC）更不可能；
- 我们的"对拍 + 显式相对容差（≤1e-4）"是正确契约，AGENTS §1.5 不需要放宽——
  但可以**分级**，让门禁更有信息量：

| 类别 | 例子 | 判据 |
|---|---|---|
| 整数/索引/离散 | scan、compaction 的索引与偏移、均匀网格的 cell/桶下标、k=16 邻接表 | **bit 精确（`== 0`）**，不得给容差 |
| f32 流式/累加 | add/saxpy/gemm/reduce | 相对容差（现有 1e-4，实测 rel 1e-6~3.7e-6） |
| 顺序敏感归约 | reduce sum（累加顺序） | 记录"同 device+driver+grid 可复现"，**不承诺跨厂商** |

另外：`grid_count`/`grid_scatter` 用**整数** `atomicAdd` —— 计数与槽位分配是顺序无关的，
所以"网格构建"天然落在 bit 精确类，这在设计上是我们的优势，值得写进 `determinism.zig` 的设计文档。

---

## 7. 后续动作（按 性价比 排序）

1. **（做）adapter 选择 + probe 透出 adapter 信息**，加 `--adapter {auto|high-perf|low-power}`
   消融；回填 README 的 dGPU 数字（含"链式优势 2.69x vs 1.39x"这一条）。
2. **（做）确定性分级门禁**：把整数/索引类断言从"容差"改为"精确相等"，写入 §7.1 与测试。
3. **（写）把本文件作为"新前端准入条件"的依据**，链到 `node-system-migration.md` §5.6 / §7.1。
4. **（可选）给 Zig 上游报告两个崩溃**（都是最小复现，5 行以内）：
   - `@workGroupId(0) * @workGroupSize(0)` 在 `amdgcn-amdhsa-none` 上 panic；
   - `var x: T addrspace(.shared)` 在 SPIR-V 目标上 panic；
   外加 `std.gpu.executionMode()` 因 assembler 拒绝 `OpExecutionMode` 而不可用。
   （这三条正好是博客 "You tell!" 邀请的反馈。）
5. **（暂缓）Zig kernel 前端 spike**：等 §4.2 的准入条件至少满足"能发 LocalSize + 能声明
   binding + 有 shared/atomic"再评估；届时消融设计 = 同一 saxpy/scan 分别用 WGSL 与
   Zig-SPIR-V 出模块，比 ①`spirv-val` 通过率 ②我们 AMD iGPU 上的端到端时间 ③与 CPU 参考的 `max|diff|`。

---

## 8. 未做的验证（诚实声明）

- 没有在 NVIDIA 上跑 Vulkan-compute（我们的 wgpu-native 路径在 dGPU 上只测了 GEMM/chain/reduce 三种，
  未测 spatial；dGPU 数值与 iGPU 一致：GEMM `max|diff| = 0`，chain `rel 3.70e-6`）。
- 没有实测 D3D12/DXC 与 RADV 之间的浮点差异（§6 的跨厂商结论来自规范文本，不是本机测量）。
- 没有把 Zig 的 SPIR-V 输出喂给 `wgpuDeviceCreateShaderModule` 做端到端验证
  （按 §3.1 它连 `spirv-val` 都过不了，没到那一步）。
- AMDGCN 崩溃未最小化到具体 AIR 指令（只定位到"两个 workgroup 内置量相乘"）。

---

## 9. 附录：§5 用的 adapter 探针

两个小程序放在 `/tmp` 下编译，**不改仓库一行代码**（先把数据拿到，再决定要不要真的加功能）：

```c
// adapter-probe.c：分别用 NULL / Undefined / LowPower / HighPerformance 请求 adapter，
// 打印 WGPUAdapterInfo（description / vendor / device / backendType / adapterType /
// vendorID / deviceID）。用回调 + wgpuInstanceProcessEvents 轮询推进，与 context.zig 同模式。

// force-hp.c → force-hp.so：LD_PRELOAD 拦 wgpuInstanceRequestAdapter，强制
// powerPreference = HighPerformance（options 为 NULL 时自建一份），其余参数透传。
WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, WGPURequestAdapterOptions const* options,
                                      WGPURequestAdapterCallbackInfo cb) {
    static request_fn real;
    if (!real) real = (request_fn)dlsym(RTLD_NEXT, "wgpuInstanceRequestAdapter");
    WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    if (options) opts = *options;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    return real(instance, &opts, cb);
}
```

```bash
# 编译（头文件/库用仓库 vendor/）：
gcc -o probe adapter-probe.c -I vendor/wgpu-native/include \
    -L vendor/wgpu-native/lib -lwgpu_native -Wl,-rpath,$PWD/vendor/wgpu-native/lib
gcc -shared -fPIC -o force-hp.so force-hp.c -I vendor/wgpu-native/include -ldl
# 跑：
LD_PRELOAD=/tmp/force-hp.so ./zig-out/bin/computeAccel --kernel chain --chain pipeline --chain-lens 64
```

两种用途：探针（§5.1 表）确认"默认到底选了哪块卡"；LD_PRELOAD 用于在不改仓库代码的前提下
拿到 §5.2 的 dGPU 数字。真正实现时应在 `context.zig` 里把偏好做成参数（§5.3）。
