# 依赖（pins）

版本 pin 在 `deps/pins.env`（唯一事实来源），拉取脚本 `tools/fetch_deps.sh`。

| 依赖 | 版本 | 用途 | 存放 | 入库 |
|---|---|---|---|---|
| wgpu-native | `v29.0.1.1` | native 端 WebGPU 实现（Rust，产出 `libwgpu_native.so` + `webgpu.h`） | `vendor/wgpu-native/` | ✗（16MB zip，脚本拉取） |
| emdawnwebgpu | `v20260911.162847` | browser 端 WebGPU 实现（Dawn 的 emscripten port） | remote port 7KB `deps/emdawnwebgpu.remoteport.py`；真实包由 emcc 下到 `.em-cache/` | ✓（只入库 remote port 文件） |

## 为什么两边各用一个实现

两个实现都提供 **同一份 `webgpu.h` C ABI**，所以 `src/gpu/webgpu.zig` 里的绑定只有一份：

```
src/gpu/webgpu.zig  ──┬── native : link libwgpu_native.so
                      └── wasm32 : 符号由 emcc --use-port=deps/emdawnwebgpu.remoteport.py 提供
```

`tools/check_abi_drift.sh` 会逐字比对两个头文件里 compute 子集（27 个符号）的函数原型，
当前状态：**全部一致**（native 6766 行 / emdawn 2945 行头文件）。
**升级任一依赖后必须重跑该脚本**，否则会出现只在单一目标上崩溃的 ABI 错位。

## 两端实现差异（绑定层需要绕开的）

| 差异 | 处理 |
|---|---|
| `wgpuInstanceWaitAny` | native：wgpu-native v29 里是 `unimplemented!()`（panic）；browser：无 ASYNCIFY 时 `abort()`。**两端都不用**，统一用 `wgpuInstanceProcessEvents` 轮询泵 |
| `wgpuDevicePoll` | 只有 wgpu-native 有 → 不用 |
| `wgpuDeviceCreateShaderModuleSpirV` | 只有 wgpu-native 有；浏览器明确拒绝 SPIR-V（`"ShaderSourceSPIRV requested, but not supported in Wasm"`）→ **shader 统一用 WGSL** |
| `WGPUNativeFeature_*` / `WGPUInstanceExtras` | wgpu-native 私有 → 不用 |

## 更新流程

```bash
# 1) 改 deps/pins.env 里的版本号
# 2) 重新拉取 native 依赖
rm -f vendor/wgpu-native/.pinned-version && tools/fetch_deps.sh
# 3) 更新 emdawnwebgpu remote port（下载对应 release 的 remoteport.py 覆盖 deps/ 下的文件）
curl -Lfo deps/emdawnwebgpu.remoteport.py "<EMDAWNWEBGPU_REMOTEPORT_URL>"
rm -rf .em-cache/ports/emdawnwebgpu.remoteport*
# 4) 触发一次 emcc 拉包 + 跑 ABI 漂移检查
emcc --use-port=deps/emdawnwebgpu.remoteport.py:help
tools/check_abi_drift.sh
# 5) zig build test && zig build wasm
```
