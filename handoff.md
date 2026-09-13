# handoff — 研究：GPU 计算后端选型（luna）

## 背景
一个 Zig 0.16 的计算加速库（`tmp/computeAccel`），`BackendType` 已把 GPU 后端留作占位（`gpu_webgpu` / `gpu_cuda`）。现在要为 GPU 加速做**选型决策**。

## 目标（纯咨询/分析，不写代码）
在以下候选里，评估哪个最适合「给 GPU 做计算加速」，评估维度**优先级**：
1. **兼容性**（AMD / NVIDIA / Intel，以及**国产 GPU**：摩尔线程、壁仞、天数智芯、兆芯、海光、沐曦等）
2. **环境配置难度**
3. **实现难度**（在 Zig 里对接、kernel 语言、样板代码量）

候选：
- Vulkan Compute（SPIR-V / GLSL-HLSL 编译）
- WebGPU（wgpu-native / mach-gpu / **Zig 0.16 std.gpu**，底层 Linux 走 Vulkan）
- OpenCL
- CUDA / OptiX（注：OptiX 只做光追，说明即可）
- HIP（AMD）、**oneAPI / SYCL**

## 需要你产出
一份结构化的**对比矩阵 + 明确推荐**。每个后端给出：
- 跨厂商覆盖率（AMD/NVIDIA/Intel/国产 的粗粒度量表：✓ 原生强 / ~ 依赖第三方 / ✗ 无）
- 环境配置难度（低/中/高 + 一句话）
- 实现难度（低/中/高 + 一句话 + 在 Zig 里的可行性）
- 生态/长期性一句评论

重点回答：
1. **只看「最广兼容 + 配置/实现成本可控」**，选谁？给一个主推荐 + 一个兜底推荐。
2. **国产 GPU** 的现状：它们普遍优先提供哪类驱动/API？（Vulkan？CUDA-like？OpenCL？）这对选型影响最大的是什么？
3. **Zig 0.16 std.gpu** 是不是 WebGPU 绑定？它对 compute（compute pipeline / dispatch / shader / buffer mapping）支持到什么程度？和 mach-gpu / wgpu-native 比，谁更省事？
4. 结合你已有的 `ComputeEngine(comptime BackendType)` comptime 派发架构，GPU 后端该怎么落地最适合（直接 wgpu-native、走 std.gpu、还是裸 Vulkan）？给出理由。

## 诚实声明
- 你**无法联网**验证实时驱动/版本信息，基于训练知识给出判断即可，但请**标注哪些是推测**。
- 不要在无把握处编造具体版本号/厂商 API 名称；把握不足就写「需实测验证」。

## 汇报结构
1. 一句话结论（推荐方案）
2. 逐个后端简评（按上面四维）
3. 对比矩阵表格
4. 国产 GPU 影响分析
5. 与 Zig `ComputeEngine` 架构落地的建议 + 理由
6. 风险 / 需实测验证点
保持精炼，中文输出。
