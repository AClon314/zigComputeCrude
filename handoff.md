# handoff — T2：computeAccel 浏览器（wasm）WebGPU 后端（emdawnwebgpu）

> 本文件是**当前任务**的交接文档。T1（native wgpu-native 后端）已完成并提交（`456e327`），
> 其任务文档见 `git show 1cb446c:handoff.md`；上游研究见 `docs/gpu-backend-research.md`；
> 依赖见 `deps/README.md`。后续任务：T3 = 能力探测 + 回退 + bench 集成。

## 0. 一句话

让 `gpu_webgpu` 后端**再编译出一个 wasm 目标**，在浏览器里跑通同一套 WGSL kernel，
页面自证「GPU 结果 == CPU 结果」。**复用 T1 的 `src/gpu/webgpu.zig` 绑定，不许 fork 一份。**

## 1. 关键原理（设计基石，别绕开）

native 与 browser 两个实现提供**同一份 `webgpu.h` C ABI**：

```
src/gpu/webgpu.zig ──┬── native  : link libwgpu_native.so          （T1 已通）
                     └── wasm32  : 符号由 emcc --use-port=… 提供     （本任务）
```

`tools/check_abi_drift.sh` 已验证两端 compute 子集 27 个符号原型逐字一致（含 `wgpuInstanceProcessEvents`）。
所以 **T2 原则上不需要改 `src/gpu/*.zig` 的绑定与逻辑**，只需要：

1. `build.zig` 加 wasm 目标（照 ouo 的 `addWasmStep` 抄）
2. 新增 wasm 入口 + shell.html + JS 驱动
3. 处理少数平台差异（见 §3）

## 2. 参照实现（ouo 已跑通，直接读这三份文件）

| 文件 | 读什么 |
|---|---|
| `/home/n/document/code/ouo/build.zig` 的 `addWasmStep` | Zig 编 `wasm32-freestanding` 对象 → emcc 链接 → 安装到 `zig-out/webgpu/` |
| `/home/n/document/code/ouo/src/abi/wasm.zig` | wasm 入口写法（只有 `extern` 声明，无 `@cImport`） |
| `/home/n/document/code/ouo/src/bindings/web/shell.html` | 平台闸门（Firefox 非 Windows 必须跳过）、`Module` 配置、启动流程 |
| `/home/n/document/code/ouo/src/bindings/web/wasm_main.c` | C `main()` 引用 Zig 导出（规避 emcc `EXPORTED_FUNCTIONS` 符号匹配坑） |

ouo 的 emcc 参数（我们**去掉 `-sUSE_SDL=3`**，compute 不需要窗口/SDL）：

```
emcc --use-port=deps/emdawnwebgpu.remoteport.py \
     -sALLOW_MEMORY_GROWTH=1 --closure=1 -O3 -o <out>.js \
     <zig_obj.o> src/bindings/web/wasm_main.c
```
且 `EM_CACHE=<project>/.em-cache`（必须设，否则每次重下 port）。

## 3. 硬性约束（两端差异，必须遵守）

1. **不要用 `wgpuInstanceWaitAny`**：浏览器端无 ASYNCIFY 时直接 `abort()`（我已在更新后的
   emdawnwebgpu v20260911 源码里复核：`library_webgpu.js:740`）。统一用 `AllowProcessEvents` 回调 + `wgpuInstanceProcessEvents` 轮询泵——**T1 已经就是这么写的，沿用即可**。
2. **不要开 `-sASYNCIFY`**（体积/性能代价大且我们不需要）。
3. **shader 只能 WGSL**：浏览器端明确拒绝 SPIR-V（`webgpu.cpp:1670`），沿用 T1 的 `src/gpu/shaders/*.wgsl`。
4. **不要用 `wgpuDevicePoll`**（emdawnwebgpu 没有该符号）。
5. wasm 目标**不能** link `wgpu_native`（`build.zig` 必须按 target 分支；emcc 侧提供符号）。
6. 回调 `callconv(.c)` + `userdata1` 传上下文；`WGPU_*_INIT` 宏不可用，手写 `.chain = .{ .sType = … }`。
7. 浏览器里 **buffer size 必须是 4 的倍数**，`mapAsync` 的 offset/size 有对齐要求。
8. `navigator.gpu` 是 `[SecureContext]`：需要 **https 或 localhost**。
9. Firefox 在 Linux 上不支持 WebGPU 且会拖垮浏览器 → **照抄 ouo shell.html 的 UA 闸门**，
   不支持时显示提示而不是崩溃。

## 4. 交付物

```
src/abi/wasm.zig                    # wasm 入口：export fn ca_wasm_main() 等
src/bindings/web/wasm_main.c        # C main() → 调用 Zig 导出
src/bindings/web/shell.html         # 平台闸门 + 结果展示 + 驱动 JS
build.zig                           # 新增 wasm step（-Dtarget 分支，不影响 native）
README.md                           # 浏览器构建/运行/验证说明
```

建议的导出面（越小越好）：

```
ca_wasm_main() -> void              # C main 调用；建 context、跑自证、写结果到状态字符串
ca_wasm_status() -> [*:0]const u8   # 页面轮询/读取的状态文本（"loading" / "GPU add: MATCH (...)" / 错误）
```
也可以让 Zig 调 JS（`extern fn ca_js_report(ptr, len)`），但**先做最简单能验证的路径**。

### 自证逻辑（这是本任务的核心验收，不能省）

页面里必须**同时**跑 CPU 与 GPU 的 `add`（size 建议 `1<<20`），逐元素比较后把结论写到状态文本：

```
"GPU add: MATCH (n=1048576, gpu=xx ms, cpu_simd=yy ms)"
"GPU add: MISMATCH at i=..." 
```
纯打印"跑起来了"不算通过。

## 5. 验收标准（必须自己实测 + 贴证据）

1. `zig build test` 仍全绿（native 不受影响，8/8）。
2. `zig build wasm` 成功，产出 `zig-out/webgpu/{*.js,*.wasm,shell.html}`（列出实际文件名与体积）。
3. 浏览器端自证通过：**最好用 `agent-browser` 打开页面抓状态文本**；若环境不允许（headless Chrome
   的 WebGPU 可能不可用），明确写出「需用户手动用 Chrome 打开 https://localhost:PORT/shell.html」，
   并说明你**已经验证到什么程度**（例如：wasm 能实例化、能建 device、能 dispatch——用 JS console 输出证明）。
4. 说明如何起服务（`python3 -m http.server` 在 127.0.0.1 也算 secure context；若用 https 见 ouo 的
   `bunx serve` + mkcert 方案）。
5. `git commit`（中文提交信息，含实测证据）。

> 诚实要求：如果浏览器端某步没跑通，**不要伪造**。写清卡在哪、错误信息、下一步猜想。

## 6. 不要做

- 不要改 `src/gpu/webgpu.zig` / `context.zig` / `pipeline.zig` 的**绑定与语义**（除非发现真正的平台差异，
  那时要在代码注释里写清"native/browser 差异"）。
- 不要改 CPU 后端与已有 native 测试的判定。
- 不要动 `deps/`（依赖已 pin 到最新 v20260911.162847）、不要动 `.em-cache`（已 gitignore）。
- 不要引入第三方 Zig 依赖。

## 7. 有用命令

```bash
zig version                    # 0.16.0
emcc --version                 # 6.0.9-git（pixi global wasm 环境）
ls .em-cache/ports/            # emdawnwebgpu.remoteport 已下载好（v20260911）
zig build test && zig build wasm
python3 -m http.server 8080    # 在 zig-out/webgpu/ 下起（localhost = secure context）
```
