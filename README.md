# computeAccel — CPU / native WebGPU 计算后端 demo

一个 Zig **0.16.0** 的最小示例，演示手动或智能选择计算后端，并把
`add` / `saxpy` 接到本机 GPU。CPU 后端仍由 `ComputeEngine(comptime BackendType)`
静态派发；native GPU 后端使用 wgpu-native 的 `webgpu.h` C ABI 和 WGSL。

## 支持的后端

| BackendType | 实现状态 | 说明 |
|---|---|---|
| `cpu_scalar` | ✅ | 标量循环（兜底） |
| `cpu_simd` | ✅ | `@Vector(8, f32)` SIMD |
| `gpu_webgpu` | ✅ native | wgpu-native v29.0.1.1；WGSL `add` / `saxpy`，运行时失败回退 CPU |
| `gpu_cuda` | ⬜ | T1 未实现 |

T1 只覆盖 native Linux wgpu-native；wasm/browser 后端属于后续 T2，本目录的
native 改动没有引入 emcc 或浏览器路径。

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
  webgpu.zig        # 手写 extern C ABI；无 @cImport
  context.zig       # instance/adapter/device/queue、ProcessEvents pump、错误域
  pipeline.zig      # WGSL pipeline / bind group / buffer cache、dispatch、readback
  shaders/add.wgsl
  shaders/saxpy.wgsl
```

每次 GPU 运算的数据路径是：

```text
queueWriteBuffer(host -> storage)
  -> compute pass (workgroup_size(64))
  -> queueSubmit
  -> copy output -> MapRead staging
  -> bufferMapAsync + wgpuInstanceProcessEvents pump
  -> getMappedRange -> memcpy -> unmap
```

pipeline、pipeline layout、bind group layout，以及当前 kernel/大小对应的 storage、
params、staging buffer 和 bind group 都缓存在 `GpuContext` 中，不会在每次 `add`
调用时重建 shader pipeline。越界 invocation 由 WGSL 中的 `arrayLength` 检查挡住。
当前 GPU API 对外支持 `f32`；其它 `T` 会走正确的 CPU SIMD fallback。

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

每次更新绑定后运行 ABI 漂移检查：

```bash
tools/check_abi_drift.sh
# OK: 28 compute symbols identical
```

## 选择逻辑

- `manual`：`--backend <name>` 显式选择。
- `heuristic`：`size >= 1024` 选择 `cpu_simd`，否则 `cpu_scalar`。
- `benchmark`：`--auto` 实测已经实现的后端；GPU 初始化/执行失败仍保留 CPU 回退。
