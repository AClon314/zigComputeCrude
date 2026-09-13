# AGENTS.md — 开发者与贡献规范（本仓库的唯一开发契约）

> 面向修改本仓库的人/agent。调用者文档看 `README.md`；架构背景与后续路线看
> `docs/node-system-migration.md`。本文件是**规则**，不是路线图：规则可以加，
> 但不要在没有实测理由的情况下放宽。

## 0. 项目定位

可移植计算内核 + 常驻链式运行时：CPU SIMD 与 WebGPU 共用一套内核定义，
一份 C ABI/一份 WGSL 同时编 native（wgpu-native）与浏览器（emdawnwebgpu）。
**这不是渲染器、不是 BLAS、也不是节点系统**；它是节点系统/图像管线/几何管线下面那一层。

**范围边界**：只做「与领域无关的运行时 + 并行原语 + 通用 op-graph 调度」。
Blender/节点的语义（域模型、属性传播、节点图求值、色彩/去噪策略）属于中间件，
不进本仓库。新原语要进库，必须同时满足：语义不引用具体消费方 + 有明确 workload
需要（按需驱动，不预埋；spatial/BVH 这类属于"按需候选"而非里程碑）。

## 1. 硬性不变量（改坏了直接回滚）

1. **一绑定两目标**：`src/gpu/webgpu.zig` 的手写 C ABI 子集必须同时兼容
   `vendor/wgpu-native` 与 `.em-cache/.../emdawnwebgpu` 的 `webgpu.h`。
   新增/修改符号后必须跑 `tools/check_abi_drift.sh`；扩展绑定要同步更新检查脚本的符号清单。
2. **禁用清单**：`wgpuInstanceWaitAny`、`wgpuDevicePoll`、SPIR-V（shader 只有 WGSL）、
   ASYNCIFY、push constants/immediates。异步只能 `ProcessEvents` 轮询推进。
3. **错误语义**：GPU 失败一律 `error.GpuError`；`ComputeEngine(.gpu_webgpu)` 可回退 CPU，
   但**必须**记录 `lastFallbackReason()`；CLI/bench 不得把 CPU 回退时间当作 GPU 时间。
4. **读回等待用墙钟超时**（当前 30s），超时必须 `wgpuBufferUnmap` 取消挂起 mapping——
   否则下一次 `wgpuQueueSubmit` 会触发 wgpu-native 的 fatal `buffer is still mapped`
   （abort，error scope 捕不到）。
5. **对拍是契约**：任何新内核/新后端都必须有 CPU 参考实现，并在测试或 CLI 里逐元素对拍；
   浮点用**明确的容差**（相对容差见下），不得放宽到"看起来差不多"。
6. **不 kill 任何进程**（包括 pi）；长任务用后台 + 轮询。
7. **不伪造实测结论**：性能/正确性数字必须来自真实运行，且说明机器是否独占、是否取中位数。

## 2. 设计原则

1. **消融优先（ablation-first）**：新增抽象/优化前先定义对照实验。M0 的范例是
   `--kernel chain`：三种模式（per_call / per_submit / chained）计算完全相同，
   只改变往返次数；只有对照出来的收益才值得留在代码里。
2. **控制工程复杂度**：先加最小的东西；不引入未经验证的通用机制
   （分配器、视图/stride、indirect、纹理只有在具体 workload 需要时才做）。
3. **诚实报告**：给出 CPU/GPU 双方数字与容差；GPU 输给 CPU 就写输（例如 reduce 在
   共享内存 iGPU 上只能打平）。
4. **零环依赖**：模块 import 必须无环；跨层只走公开 API。
5. **薄封装优先**：CPU SIMD 统一走 `backends` 薄层（`std.simd.suggestVectorLength` +
   `@mulAdd`），kernel 不直接散落 `@Vector(8, T)`。

## 3. 模块边界（现状 → v2 目标）

**发布形态**：一个包，多个 module（`build.zig` 的 `addModule`）：

| module | 归属内容 | 规则 |
|---|---|---|
| `computeAccel` | CPU 内核、GPU 后端、runtime、通用原语（`primitives/`）、探测/选择 | wgpu 链接可用 `-Dwebgpu=false` 关掉；不得 import `computeAccel_spatial`；新 kernel 一律走 runtime，不新增 kernel 内建 cache |
| `computeAccel_spatial` | 领域无关的空间原语（S1）：均匀网格索引（GPU build+query，CPU 参考对拍） | 只能依赖 `computeAccel` 的公开 API；不得被主 module 反向 import；GPU 布局保持 WebGPU 基线（≤8 storage binding/阶段，AoS vec4 是为此的取舍） |

新增一类功能（图像、物理、其他域）= **新增一个 module**，不要塞进主 module。
每次新增 module 都要同步 `tools/tree_shake_probe.zig` 的断言清单（保证"未引用即不编译"）。


现状（可直接使用）：

```
src/root.zig          # 公共入口（窄 API + 测试聚合）
src/backend.zig       # BackendType / SelectionMode / heuristic 闸门
src/engine.zig        # ComputeEngine(comptime bt)：CPU scalar/simd 内核
src/bench.zig         # 计时 + pickBest（端到端选择，不含 GPU 失败时间）
src/buffer.zig        # DeviceBuffer stub（历史 API；新代码用 runtime.Buffer）
src/gpu/webgpu.zig    # C ABI 子集（唯一允许写 extern 的地方）
src/gpu/context.zig   # device/queue/limits/probe/pump + 共享 plumbing（pipeline/buffer/submit/readback）
src/gpu/pipeline.zig  # add/saxpy（旧形态：每调用一次往返）
src/gpu/gemm.zig      # GEMM simple/tiled + CPU 参考 + 对拍测试
src/gpu/reduce.zig    # reduce sum/max + CPU 参考 + 对拍测试
src/runtime*.zig      # M0：Buffer / Kernel / Chain（常驻 + 链式）
src/chain_bench.zig   # M0 消融基准（saxpy 链、GEMM→bias→reduce 链）
```

v2 目标布局（迁移路径见 `docs/node-system-migration.md` §6）：`runtime/`、
`backends/{cpu,webgpu}/`、`primitives/`、`determinism.zig`、`capability.zig`。
迁移规则：**旧的按 kernel 划分的 shape-keyed cache 逐步并入 runtime.Buffer**，
新内核一律走 `runtime.Kernel` + `Chain`。**全部内置 kernel 已迁移**：add/saxpy（`primitives/elementwise.zig`；
`gpu/pipeline.zig` 是兼容 shim）、gemm（`Kernels` + `VariantCache`）、
reduce（`Kernels` + `bindAll`/`Binding`）；旧的内建 cache 已从 `GpuContext` 删除。

依赖方向（无环）：`root` → 各模块；`runtime` → `gpu/context` + `gpu/webgpu`；
`chain_bench` → `runtime` + 参考实现；`gpu/*` 不反向依赖 `runtime`。

## 4. 新增一个内核的流程（checklist）

1. `src/gpu/shaders/<name>.wgsl`：定好 entry point 与 binding 顺序/类型/访问模式；
   参数用 16B uniform（无 push constants），2D 展平下标写 `num_workgroups.x * workgroup_size`。
2. Zig 侧二选一：
   - 新形态（推荐）：`runtime.Kernel.init(ctx, shader, entry, bindings, workgroup_size)`，
     调用方用 `runtime.Buffer` 管理数据；
   - 旧形态：在 `src/gpu/<name>.zig` 按 gemm/reduce 的样子实现（必须在 README 标注为 legacy）。
3. CPU 参考实现（scalar + 可选 SIMD）放在同文件，供测试/对拍/CLI 共用。
4. 测试（`zig build test` 必须全绿；无 GPU 时用 `error.SkipZigTest`）：
   - CPU scalar vs SIMD 在边角 shape 上一致；
   - GPU vs CPU 逐元素对拍：**浮点相对容差 ≤ 1e-4**（如需更松必须在代码里写理由）；
   - 边界：0/1/非整块/超 limits。
5. 需要 CLI/基准时在 `main.zig` 增加入口；性能数字回填 README。
6. 改动了绑定 → `tools/check_abi_drift.sh`；改了 wasm 路径 → `zig build wasm`。

## 5. 测试与验收门槛（缺一不可）

```bash
zig build test --summary all     # 全绿（含 ABI 检查与 tree-shake 门禁；GPU 测试允许 Skip）
zig build wasm                   # 成功（浏览器路径不回归）
tools/check_abi_drift.sh         # OK
zig build tree-shake             # CPU-only 消费者产物无 wgpu/WGSL/spatial 符号
zig build consumer-check         # examples/cpu_consumer 作为 path 依赖可构建可运行
```

tree-shake 门禁有正负两侧：`build.zig` 里的 probe 只调用 CPU 路径（必须过）；
负例可直接用手工构造的含 GPU 对象验证脚本确实会失败
（`tools/check_tree_shake.sh <object-with-gpu-code>` 必须 exit 1）。

- 改 GPU 路径后要跑一遍真实 CLI/消融（`--kernel gemm|reduce|chain`），
  确认没有回退、验证 `max|diff|`/`rel diff`。
- 测试里的容差必须显式写出量级；不得为了让测试过而放宽既有断言。

## 6. 依赖决策（不要随手替换）

| 依赖 | 决策 | 理由 |
|---|---|---|
| Zig `@Vector` + `std.simd` | 保留（薄封装） | 零依赖、跨目标（含 wasm simd128）；实测 CPU 侧受内存带宽限制，SIMD 指令集不是瓶颈 |
| wgpu-native（desktop） | 保留 | 稳定的 `webgpu.h` C ABI、预编译、覆盖 Vulkan/Metal/DX12，且已有漂移检查 |
| emdawnwebgpu（browser） | 保留 | 唯一现实的浏览器 WebGPU + 同形 C ABI |
| 原语（scan/sort/BVH/texture…） | 自研 | 无跨 desktop+browser 的现成可移植库（详见 docs §5） |
| C/C++ 领域库（OIIO/OCIO/OIDN/FastNoise2…） | 按需、CPU 侧 | 强在 I/O/色彩/去噪；不进 WebGPU 核心 |
| wgpu-native 二进制 | vendored（默认）或 `-Dwgpu-lib-dir` | `vendor/` 不入包（不入库），发布形态由消费者指路 |
| Halide / naga / Dawn(native) / SPIR-V | 备选 | 只有放弃浏览器或要行为对齐时才评估 |

## 7. 提交与工作方式

- 一个任务一个 commit，中文 message，**带实测数字或验收命令**；
- 提交前工作区自检：`git status` 干净、README/AGENTS 与实际一致；
- `*handoff*` 是本地工作日志（已 gitignore），不入库；任务背景写在那里或 `docs/`；
- 大改前先写/更新 `docs/` 里的评估与消融设计，再落代码。

## 8. 已知坑（别重复踩）

- **`vfmadd` 不会自动出现**：Zig 严格浮点不收缩 `a*b+c`，要用 `@mulAdd`。
- **位宽不要写死**：用 `std.simd.suggestVectorLength(T)`。
- **readback 超时**：见 §1.4；固定"泵次数"当超时是错的（与排队 GPU 工作量不成比例）。
- **uniform 布局**：WGSL uniform struct 按 16B 对齐/补齐（例如 `{f32}` 也要 16B buffer）。
- **2D dispatch**：单轴上限 65535，任何新 kernel 都要走 `workgroupGrid` 的 limits 闸门。
- **浏览器没有 WaitAny**：不要写阻塞式新 API；runtime 的 `Chain.submit` 是"提交+阻塞回读"，
  在浏览器端需要由 rAF pump 驱动（移植时注意）。
- **测量失真**：纯归约会被 LICM 提出循环（`timeReduceCpu` 每轮扰动一个元素）；
  基准先 warmup 再计时（`--kernel chain` 已内置）。
- **indirect buffer 的 usage 排他**：同一个 dispatch 里，buffer 不能既作为
  `dispatchIndirect` 的来源又绑定为 storage（wgpu usage scope 报
  "STORAGE_READ_WRITE ... cannot be used with ... INDIRECT"，且 `wgpuQueueSubmit`
  对 validation error 直接 abort）。做法：写 count 与间接派发拆成两次 dispatch、
  两个 shader（参见 `indirect_probe.wgsl` / `indirect_fill.wgsl` 与
  `Chain.dispatchIndirect` 的测试）；control buffer 用 `runtime.buffer.indirect`。
- **`Chain.wait()` 必须被调用**：`submitAsync()` 只提交，不调用 `wait()` 时
  `download()` 的 out 切片不会被写入（数据仍在 device 上），`deinit()` 也不会自动
  `wait`；每个 submit 路径都要配一次 wait（`submit()` = 两者合一）。
- **scan 的块内分配必须连续**：单 workgroup 扫块前缀和时，每个线程要拿**连续**块
  （`[lid*chunk, (lid+1)*chunk)`），跨步分块会破坏前缀顺序（已在
  `scan.wgsl::block_scan_global` 修正）。
