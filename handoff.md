# handoff — T1：computeAccel 原生 WebGPU 后端（wgpu-native）

> 本文件是**当前任务**的交接文档。上游研究结论见 `docs/gpu-backend-research.md`；依赖见 `deps/README.md`。
> 规划中的后续任务：T2 = browser/wasm 后端（emdawnwebgpu），T3 = 能力探测 + 回退 + bench 集成。**本任务只做 T1。**

## 0. 一句话

用 wgpu-native 的 C ABI 在 Zig 里实现 `gpu_webgpu` 后端，让
`ComputeEngine(.gpu_webgpu).add/saxpy` 在本机 GPU 上跑出与 `cpu_simd` **逐元素一致**的结果，
并打印实测吞吐。**不引入任何第三方 Zig 包**，绑定自己写。

## 1. 环境事实（已实测确认，不要再怀疑）

| 项 | 值 |
|---|---|
| Zig | `0.16.0`（pixi global，`zig version` 可验） |
| 依赖 | `vendor/wgpu-native/`（v29.0.1.1 预编译，`tools/fetch_deps.sh` 已拉过） |
| 头文件 | `vendor/wgpu-native/include/webgpu/webgpu.h`（6766 行）+ `webgpu.h`(同目录 `wgpu.h` 是 native 私有扩展) |
| 库 | `vendor/wgpu-native/lib/libwgpu_native.so`（9.3MB）+ `libwgpu_native.a` |
| 本机 GPU | AMD Radeon Vega iGPU（radv/Vulkan）+ NVIDIA RTX 3050 Mobile；`vulkaninfo` 可用 |
| 已跑通的 C 探针 | `wgpuCreateInstance` → `wgpuInstanceRequestAdapter` → `wgpuAdapterRequestDevice` → `wgpuDeviceGetQueue` → `wgpuDeviceCreateBuffer` **全部返回非空**；`wgpuAdapterGetInfo` 报 `backendType=Vulkan(6), vendorID=0x1002(AMD)` |

C 探针用的调用姿势（**照抄这个模式**，本任务就是用 Zig 复刻它）：

```c
wgpuInstanceRequestAdapter(inst, NULL,
    (WGPURequestAdapterCallbackInfo){ .mode = WGPUCallbackMode_AllowProcessEvents,
                                      .callback = on_adapter, .userdata1 = &adapter });
for (int i = 0; i < 200 && !adapter; i++) wgpuInstanceProcessEvents(inst);   // ← 轮询泵
```

## 2. 硬性约束（踩过的坑，别踩回去）

1. **绝对不要调用 `wgpuInstanceWaitAny`**。
   - native：wgpu-native v29 里它是 `src/unimplemented.rs` 里的 `unimplemented!()` → Rust panic → `abort()`。**我实测崩了**。
   - browser（T2）：emdawnwebgpu 无 ASYNCIFY 时同样 `abort('TODO: Implement asyncify-free WaitAny for timeout=0')`。
   - 唯一两端通用的推进方式：回调用 `WGPUCallbackMode_AllowProcessEvents` + 循环 `wgpuInstanceProcessEvents(instance)`。
   - 绑定层统一暴露 `pump()`（内部就是 ProcessEvents 循环），**不要**把 waitAny 暴露给上层。
2. **不要用 `wgpuDevicePoll`**（emdawnwebgpu 里没有这个符号，用了绑定层就不通用了）。
3. **shader 只用 WGSL**。浏览器端 Dawn 明确拒绝 SPIR-V（`"ShaderSourceSPIRV requested, but not supported in Wasm"`）。native 想用 SPIR-V 是以后的事。
4. 回调必须是 `callconv(.c)`，**上下文只能经 `userdata1` 传**（Zig 闭包不能转函数指针）。
5. C 的 `WGPU_*_INIT` 宏在 Zig 里不可用 → 手写 `.{ .chain = .{ .sType = WGPUSType_ShaderSourceWGSL } }`。`sType` 填错是运行时报错的头号来源。
6. Zig 0.16 的 std API 与老教程差别大（没有 `argsAlloc`、没有 `fs.cwd()`、`ArrayList` 没有 `.writer()`）。参考同仓库现有代码与 `build.zig` 的写法。

## 3. 交付物

```
src/gpu/
  webgpu.zig        # 最小绑定：~30 个 extern fn + 需要的 extern struct/enum。纯声明，无 @cImport。
  context.zig       # GpuContext：instance/adapter/device/queue 生命周期 + pump() + 错误域
  pipeline.zig      # shader module / compute pipeline / bind group 缓存 + dispatch + readback
  shaders/add.wgsl  # @embedFile 用
  shaders/saxpy.wgsl
```
改动：
- `src/engine.zig`：`.gpu_webgpu` 分支接上（`add` / `saxpy`），**不要再 panic**
- `src/backend.zig`：`gpu_webgpu` 的 `isImplemented()` 视情况调整（若做成运行时探测，保持 `false` 也行，见下）
- `build.zig`：native 目标链接 `vendor/wgpu-native/lib/libwgpu_native.so`，并让运行时能找到它（`addLibraryPath` + `linkSystemLibrary("wgpu_native")`；运行时用 rpath 或把 `.so` 安装到 `zig-out/lib` 并设 rpath `$ORIGIN/../lib`）
- `README.md`：更新「支持的后端」表 + GPU 运行/验证方法 + 踩坑

## 4. 绑定清单（照这个写，不要多写）

**符号**（全部在 `webgpu.h` 里；已用 `tools/check_abi_drift.sh` 验证 native 与 browser 原型逐字一致）：
`wgpuCreateInstance`、`wgpuInstanceProcessEvents`、`wgpuInstanceRequestAdapter`、`wgpuAdapterRequestDevice`、`wgpuAdapterGetInfo`、`wgpuDeviceGetQueue`、`wgpuDevicePushErrorScope`、`wgpuDevicePopErrorScope`、`wgpuDeviceCreateShaderModule`、`wgpuDeviceCreateComputePipeline`、`wgpuComputePipelineGetBindGroupLayout`、`wgpuDeviceCreateBindGroupLayout`、`wgpuDeviceCreateBindGroup`、`wgpuDeviceCreatePipelineLayout`(可省)、`wgpuDeviceCreateCommandEncoder`、`wgpuCommandEncoderBeginComputePass`、`wgpuCommandEncoderFinish`、`wgpuComputePassEncoderSetPipeline`、`wgpuComputePassEncoderSetBindGroup`、`wgpuComputePassEncoderDispatchWorkgroups`、`wgpuComputePassEncoderEnd`、`wgpuDeviceCreateBuffer`、`wgpuQueueWriteBuffer`、`wgpuQueueSubmit`、`wgpuBufferMapAsync`、`wgpuBufferGetMappedRange`、`wgpuBufferUnmap`，外加对应的 `*Release`（instance/adapter/device/queue/buffer/shaderModule/pipeline/bindGroup/bindGroupLayout/commandEncoder/computePassEncoder/commandBuffer）。

**枚举/常量**：`WGPUBufferUsage_Storage|CopyDst|CopySrc|MapRead`、`WGPUMapMode_Read`、`WGPUShaderStage_Compute`、`WGPUCallbackMode_AllowProcessEvents`、`WGPU_STRLEN`、`WGPU_WHOLE_SIZE`。
**结构体**：`WGPUStringView{data,length}`、`WGPUChainedStruct`、`WGPURequestAdapterCallbackInfo`、`WGPURequestDeviceCallbackInfo`、`WGPUBufferMapCallbackInfo`、`WGPUBufferDescriptor`、`WGPUShaderModuleDescriptor`+`WGPUShaderSourceWGSL`、`WGPUComputePipelineDescriptor`+`WGPUComputeState`、`WGPUBindGroupLayoutDescriptor`+`...Entry`、`WGPUBindGroupDescriptor`+`...Entry`、`WGPUCommandEncoderDescriptor`、`WGPUComputePassDescriptor`。

> 字段的**顺序和类型必须与 C 头文件完全一致**（extern struct）。写的时候逐字段核对头文件，不要凭记忆。

## 5. 实现要点

- **数据流**：`Upload(host→device storage buffer)` → `dispatch(workgroups)` → `submit` → `copy result buffer → staging(MapRead)` → `mapAsync` → `pump` → `getMappedRange` → 拷贝出来 → `unmap`。
- **pipeline/bind group 缓存**：不要每次 `add` 都重建 pipeline（那会把 GPU 优势吃光）；按 (kernel, buffer 数量/大小) 缓存在 GpuContext 里。
- **dispatch 尺寸**：`workgroup_size(64)`，`workgroupCountX = ceil(n/64)`；最后一次 dispatch 越界的线程必须在 shader 里用 `if (gid >= n) { return; }` 挡住。
- **readback 之后必须 `unmap`**，且 `getMappedRange` 指针只在 map..unmap 之间有效。
- **错误处理**：至少接上 `wgpuDevicePushErrorScope`/`PopErrorScope` 或 device uncaptured error 回调，把失败变成 `error.GpuError` 而不是静默错误结果。
- **失败要能回退**：`GpuContext.init()` 任何一步失败（无 GPU / 无 adapter / device 创建失败）都返回错误，调用侧回退 CPU。

## 6. 验收标准（必须自己实测，把输出贴进 commit message）

1. `zig build test` 全绿（CPU 测试行为不变）。
2. 新增 GPU 测试：`size = 1<<20` 的 `add` 结果与 `cpu_scalar`/`cpu_simd` **逐元素一致**（`expectEqualSlices`）；`saxpy` 同理。
3. `zig build run -- --backend gpu_webgpu --size 1048576 --iters 20` 能跑出正确结果（不再 panic）。
4. 打印实测吞吐（GB/s）与 `cpu_simd` 对比，并把数字写进 README / commit message。
   - 注意：**小数组 GPU 更慢是正常的**（传输+提交开销），不要为了"看起来快"去调参数；大数组（≥1<<20）应能看到 GPU 优势。
5. GPU 不可用时测试要 **skip 而不是 fail**（本机有 GPU，但 CI/他人机器可能没有）。
6. 完成后 `git commit`（中文提交信息，说明改动 + 实测数据）。

## 7. 不要做

- 不要改 `cpu_scalar` / `cpu_simd` 的行为、阈值或测试。
- 不要动 wasm/browser 相关（T2 的任务）：不要动 `deps/`、`.em-cache`、emcc 相关。
- 不要引入第三方 Zig 依赖包（`build.zig.zon` 的 dependencies 保持为空）。
- 不要用 `wgpuInstanceWaitAny` / `wgpuDevicePoll` / SPIR-V。

## 8. 有用的命令

```bash
zig version                                   # 0.16.0
tools/fetch_deps.sh                           # 依赖（已拉过，幂等）
tools/check_abi_drift.sh                      # ABI 漂移检查（改完绑定跑一次）
zig build test                                # 单元测试
zig build run -- --backend gpu_webgpu --size 1048576 --iters 20
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json zig build run ...   # 指定走 AMD
```
