# computeAccel — CPU / native + 浏览器 WebGPU 计算后端 demo

一个 Zig **0.16.0** 的最小示例，演示手动或智能选择计算后端，并把
`add` / `saxpy` 接到 GPU。CPU 后端由 `ComputeEngine(comptime BackendType)`
静态派发；GPU 后端使用 `webgpu.h` C ABI 和 WGSL，**同一份绑定与同一份 shader
同时编译到 native（wgpu-native）与浏览器（emdawnwebgpu/wasm）**。

## 支持的后端

| BackendType | 实现状态 | 说明 |
|---|---|---|
| `cpu_scalar` | ✅ | 标量循环（兜底） |
| `cpu_simd` | ✅ | `@Vector(8, f32)` SIMD |
| `gpu_webgpu` | ✅ native + browser | native: wgpu-native v29.0.1.1；browser: emdawnwebgpu v20260911.162847（wasm）；WGSL `add` / `saxpy`，运行时失败会明确标注并回退 CPU |
| `gpu_cuda` | ⬜ | 未实现 |

## 构建与运行

依赖已经放在 `vendor/wgpu-native/`（预编译的 `libwgpu_native.so`）。构建脚本会：

- 用 `addLibraryPath` + `linkSystemLibrary("wgpu_native")` 链接它；
- 把 `.so` 安装到 `zig-out/lib`；
- 给安装后的 `zig-out/bin/computeAccel` 加 `$ORIGIN/../lib` rpath，同时给测试加入 vendor rpath。

```bash
zig version                         # 0.16.0
zig build                           # 编译并安装 bin + lib
zig build test                      # CPU 测试 + native GPU 测试
zig build test -Doptimize=ReleaseFast

# 手动指定 native GPU（默认 adapter 由 Vulkan 驱动选择）
zig build run -- --backend gpu_webgpu --size 1048576 --iters 20

# 也可以指定 AMD RADV adapter；无 GPU 的机器上 GPU 测试会 skip，CLI 会回退 CPU
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  zig build run -- --backend gpu_webgpu --size 1048576 --iters 20

# CPU 对照
zig build run -- --backend cpu_scalar --size 1048576 --iters 20
zig build run -- --backend cpu_simd   --size 1048576 --iters 20
zig build run -- --auto --size 1048576
```

CLI 的吞吐定义沿用原 demo：`2 * size * sizeof(f32) * iters / elapsed`，即按两个
输入数组计算 GB/s，便于和 CPU 行直接比较。`gpu_webgpu` 行是端到端时间，包含输入
上传、命令提交、结果 copy、`mapAsync`/pump 和 readback；因此简单逐元素 kernel 在
一台带宽较低的 iGPU 上不一定超过 CPU SIMD，这个数字没有隐藏传输开销。额外的
`gpu_batch` 行把相同输入上传一次、提交 `iters` 次真实 dispatch、最后 readback 一次，
用于观察 steady-state kernel 吞吐；它明确标注为批量测量，不能冒充端到端结果。

## 本机实测（T1 验收记录）

环境：AMD Radeon Vega iGPU（RADV/Vulkan；系统同时有 NVIDIA RTX 3050 Mobile），
命令为：

```text
zig build run -- --backend gpu_webgpu --size 1048576 --iters 20
```

一次实测输出（Debug，2026-09-13）：

```text
selected backend = gpu_webgpu  (mode=manual, size=1048576, iters=20)

backend      total_ns    throughput(GB/s)
cpu_scalar   49793802   3.369
cpu_simd     11431995   14.676
speedup (scalar/simd) = 4.36x
gpu_webgpu  36060810   4.652
speedup (gpu/cpu_simd) = 0.32x
gpu_batch    10324870   16.249
speedup (gpu_batch/cpu_simd) = 1.11x
result sample: 5.0, 5.0, 5.0, 5.0 (expected 5.0)
```

本次 simple `add` 的 GPU 端到端吞吐为 **4.652 GB/s**，CPU SIMD 为
**14.676 GB/s**；GPU 比 CPU 标量高 **1.38x**，端到端仍受搬运/映射开销影响。
同一次运行的 `gpu_batch` steady-state 吞吐为 **16.249 GB/s**，相对 CPU SIMD 为
**1.11x**；这是真实执行的 20 次 GPU dispatch，但只做一次输入上传和一次结果
readback，故单独列出而不掩盖端到端数字。

正确性和 GPU 测试还实际验证了：`size = 1<<20` 的 `add`、`saxpy` 结果都用
`std.testing.expectEqualSlices` 分别和 `cpu_scalar`、`cpu_simd` 比较；本机两项
GPU 测试均通过。GPU 初始化失败（没有 adapter/device）时测试返回
`error.SkipZigTest`，而不是失败。

## native WebGPU 实现

```text
src/gpu/
  webgpu.zig        # 手写 extern C ABI；无 @cImport（native 与 wasm 共用）
  context.zig       # instance/adapter/device/queue、ProcessEvents pump、错误域
  pipeline.zig      # WGSL pipeline / bind group / buffer cache、dispatch、readback
  shaders/add.wgsl
  shaders/saxpy.wgsl
  shaders/add_source.zig   # @embedFile 桥（wasm 侧用同一份 add.wgsl）
src/abi/wasm.zig           # wasm 入口：状态机 + ca_wasm_main/pump/status
src/bindings/web/wasm_main.c   # C main() 引用 Zig 导出（emcc 符号保留）
src/bindings/web/shell.html    # 平台闸门 + rAF pump + 结果展示
```

每次 GPU 运算的数据路径是：

```text
queueWriteBuffer(host -> storage)
  -> compute pass (2D dispatch, workgroup_size(64))
  -> queueSubmit
  -> copy output -> MapRead staging
  -> bufferMapAsync + wgpuInstanceProcessEvents pump
  -> getMappedRange -> memcpy -> unmap
```

pipeline、pipeline layout、bind group layout，以及当前 kernel/大小对应的 storage、
params、staging buffer 和 bind group 都缓存在 `GpuContext` 中，不会在每次 `add`
调用时重建 shader pipeline。越界 invocation 由 WGSL 中的 `arrayLength` 检查挡住。
当前 GPU API 对外支持 `f32`；其它 `T` 会走正确的 CPU SIMD fallback。

### 限制与已知边界

WebGPU 的 `maxComputeWorkgroupsPerDimension` 是**每个 dispatch 轴**的上限，不能把
它误读成整个计算只能有 65,535 个 workgroup。本项目把线性的 workgroup 流铺成
`x = min(limit, groups)`、`y = ceil(groups / x)` 的 2D grid；`add.wgsl` 与
`saxpy.wgsl` 用 `num_workgroups.x` 和 `global_invocation_id.y` 还原线性下标，越界
保护仍由原来的 `arrayLength` 判断负责。因此 `groups = 65,536` 及更大的常见输入
不会再因为 1D dispatch 上限而回退；只要 2D grid 两轴和 buffer 大小仍在设备能力内，
GPU 路径会正常执行。

初始化时从 `wgpuAdapterGetLimits` 和 `wgpuDeviceGetLimits` 读取 limits，并把设备
实际使用的 `maxComputeWorkgroupsPerDimension`、`maxStorageBufferBindingSize`、
`maxBufferSize` 缓存在 `GpuContext` / `ProbeResult`。`GpuContext.canRun()` 同时检查
这三个边界（输入、输出和 staging buffer 都必须能创建）；超过边界的请求会在提交
前返回 GPU 错误，`heuristic` 与 `--auto` 不会把它当成可运行候选。浏览器端复用同一
份 2D 感知 WGSL；当前 demo 固定为 `1<<20`，并且 dispatch 也走同样的 2D 形式。


### ABI / 踩坑记录

- 绑定只声明 `webgpu.h` 的 C ABI 子集，回调统一是 `callconv(.c)`，上下文只经
  `userdata1` 传递。
- 异步 adapter、device、map、error-scope 回调都使用
  `WGPUCallbackMode_AllowProcessEvents`，由 `wgpuInstanceProcessEvents(instance)`
  循环推进；**不调用也不暴露 `wgpuInstanceWaitAny`**。
- 不使用 `wgpuDevicePoll`，shader 只用 WGSL，不传 SPIR-V。
- C 的 `WGPU_*_INIT` 宏不能在 Zig 中直接使用；WGSL descriptor 的 chain 手写为
  `.sType = WGPUSType_ShaderSourceWGSL`。
- staging buffer 是 `MapRead | CopyDst`，GPU output 是 `Storage | CopySrc`；readback
  后必须在 mapped range 仍有效时拷贝，随后立即 `unmap`。

每次更新绑定后运行 ABI 漂移检查（已接进 `zig build test`，合并后会当场拦住）：

```bash
zig build test        # 含漂移检查；.em-cache 未解包时打印 SKIP（仍退出 0）
zig build abi-check   # 严格模式：输入缺失即失败，依赖升级/CI 用
tools/check_abi_drift.sh            # 也可单独跑
# [abi] OK: 31 compute symbols identical
```

行为：

- 逐字比对 native（`vendor/wgpu-native/include/webgpu/webgpu.h`）与 browser
  （`.em-cache` 里解包的 emdawnwebgpu `webgpu.h`）的 compute 子集函数原型；
- 任一符号缺失或签名不同 → **`zig build test` 失败**（exit 1）；
- 两边头文件版本与 `deps/pins.env` 不一致 → 打 `WARN`（升级依赖后忘了重新解包的典型症状）；
- 新环境没跑过 `zig build wasm`（`.em-cache` 为空）→ 打 `SKIP` 并提示生成方法，不阻塞测试。

## 浏览器 WebGPU（wasm）

同一份 `src/gpu/webgpu.zig` 绑定与同一份 WGSL kernel 编译到 wasm：Zig 产出
`wasm32-freestanding` 对象，emcc 用 emdawnwebgpu port 链接并补上 WebGPU 符号。

```bash
zig build wasm                      # -> zig-out/webgpu/{computeAccel.js,.wasm,shell.html}
cd zig-out/webgpu && python3 -m http.server 8080
# 用 Chrome 打开 http://127.0.0.1:8080/shell.html （localhost 也算 secure context）
```

页面会先跑 CPU SIMD `add`，再用 GPU 跑同一份 WGSL `add`，逐元素比较 1,048,576 个
f32 并把结论写进状态文本；**只有显示 `GPU add: MATCH` 才算通过**。

实测（Chrome，2026-09-13，agent-browser 抓取页面状态）：

```text
GPU add: MATCH (n=1048576, gpu=31.000 ms, cpu_simd=0.800 ms)
```

产物体积：`computeAccel.js` 14 KB、`computeAccel.wasm` 77 KB、`shell.html` 6.6 KB。

### 浏览器端的关键约束

- **异步只能靠 pump**：不启用 `-sASYNCIFY`，因此**不能**用 `wgpuInstanceWaitAny`
  （emdawnwebgpu 在无 ASYNCIFY 时直接 `abort()`）。统一走
  `WGPUCallbackMode_AllowProcessEvents` + `wgpuInstanceProcessEvents`，浏览器由页面的
  `requestAnimationFrame` 驱动 `ca_wasm_pump()`。
- **只能用 WGSL**：emdawnwebgpu 明确拒绝 SPIR-V（`ShaderSourceSPIRV ... not supported
  in Wasm`），故 shader 只维护 WGSL 一份。
- **Firefox on Linux 不支持**：`navigator.gpu` 会暴露但初始化会拖垮浏览器
  （Mozilla bug 2006676），`shell.html` 已做 UA 闸门，直接提示改换 Chrome。
- 需要 https 或 localhost（`navigator.gpu` 是 `[SecureContext]`）。
- `zig build` 的日志里可能出现 `failed command: EM_CACHE=... emcc ...` —— 这是 Zig 0.16
  在子命令向 stderr 输出内容时的前缀噪音（emcc 的 clang 版本 warning）；**以
  `Build Summary: ... success` 与退出码为准**。

## 选择逻辑

- `manual`：`--backend <name>` 是显式请求。请求 `gpu_webgpu` 不代表一定会由
  GPU 执行；初始化或运算失败时仍保证 CPU 结果正确，并在选择行写出
  `gpu_webgpu (fell back: <reason>)`。
- `heuristic`：小于 `gpu_threshold = 1<<22` 时只在 CPU scalar/SIMD 中选择（`size >=
  1024` 为 `cpu_simd`）。达到 GPU 闸门后，先调用一次缓存的
  `GpuContext.probe()`，并同时检查 f32 字节数、2D workgroup grid 和设备 limits；
  只有探测成功且 `canRun()` 为真才返回 `gpu_webgpu`，否则返回 `cpu_simd`。
  这个阈值是根据本机 T1 实测确定的保守闸门：`size=1<<20` 时 GPU 端到端
  4.652 GB/s、CPU SIMD 14.676 GB/s，不能把“有 GPU”当成“GPU 更快”。
- `benchmark`：`--auto` 先做能力探测，然后只把探测成功的 GPU 纳入**端到端**
  `add` bench（每次都包含上传、dispatch、copy、map/readback）；GPU 错误直接从
  候选集中剔除，不会用 CPU fallback 的时间冒充 GPU。CLI 会说明 GPU 是否可用、
  实测 ns 与 `cpu_simd` 的比较，以及最终为什么保留某个后端。
- `gpu_batch` 仍是单独的 steady-state 观察项（一次上传/一次 readback、多次真实
  dispatch），不参与 `--auto`，也不能和 CPU 的端到端数字混称。

## 能力探测与回退语义

`GpuContext.probe()` 在 native 进程内用线程安全的一次性缓存建立 instance、adapter、
device 和 queue，并读取 adapter/device limits。成功与失败都缓存；失败的 `ProbeResult`
同时提供 `failure` 枚举和稳定的 `reason` 文本，例如 instance 创建失败、limits 查询
失败、adapter/device 请求失败或初始化回调超时。成功结果还带有实际设备的
`maxComputeWorkgroupsPerDimension`、`maxStorageBufferBindingSize`、`maxBufferSize`，
可用 `ProbeResult.canRun()` 预判某个请求。成功探测得到的 context 会直接供后续 GPU
调用复用，不会“探测一次、执行时再悄悄建另一个 context”。`resetGlobal()` 仅供测试或
明确要重试驱动环境的应用使用。

低层 `computeAccel.gpu.add` / `saxpy` 保留 `!void` 错误，让调用方能区分 GPU 失败；
`ComputeEngine(.gpu_webgpu)` 为兼容原有的 `void` API 仍会回退 `cpu_simd`，但会记录
原因。可通过 `computeAccel.gpu.lastFallbackReason()` 查询最近一次回退；CLI 和自动
选择不会把这条路径打印成普通 GPU 结果。对非 `f32` 类型也会明确记录“仅支持 f32”
后回退 CPU SIMD。

例如人为禁用 Vulkan ICD：

```bash
VK_ICD_FILENAMES=/nonexistent.json \
  zig build run -- --backend gpu_webgpu --size 1048576
```

输出会保留正确的 `5.0` 结果，并类似下面明确标注原因，而不是打印普通 GPU 行：

```text
selected backend = gpu_webgpu (fell back: WebGPU device request failed or returned no device) ...
result sample: 5.0, 5.0, 5.0, 5.0 (expected 5.0)
```
