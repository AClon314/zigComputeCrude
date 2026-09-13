# handoff-3steps — GPU 后端的三步路线图

> 本文是**路线图/科普文档**，给三件事做定性说明、术语对号与验收标准。
> 当前正在执行的任务写在 `handoff.md`；上游选型研究见 `docs/gpu-backend-research.md`；
> 依赖见 `deps/README.md`。
>
> 状态：**Step 2 已完成**（commit `fda10dc`）；Step 1 与 Step 3 未开始。

---

## 0. 一个类比，先把整个体系立住

| 名字 | 大白话 |
|---|---|
| CPU + 内存 | 你的书桌。工具箱就在手边，但一次只能算几件事 |
| GPU + 显存 | 隔壁的工具间。几十个工人同时干活，但材料不放在你手边 |
| 上传 / 下载（往返） | 把材料从书桌搬到工具间、再把成品搬回来。走廊速度是固定的 |

关键事实：**走廊很慢。**

实测（`add`，100 万个数逐元素相加，本机 AMD Vega iGPU）：

| 做法 | 吞吐 |
|---|---|
| 书桌上自己算（CPU SIMD） | ~13 GB/s |
| 每次搬过去算一次再搬回来（当前 GPU 路径） | **~3.5 GB/s**（比 CPU 慢 4 倍） |
| 只搬一次、在里面连算 20 遍（`gpu_batch` 测量） | ~13 GB/s（打平） |

结论：**现在 GPU 输，不是输在算得慢，是输在搬运。** 搬运时间里真正计算的可能只占 5%。

---

## 1. 总览

| Step | 主题 | 状态 | 一句话目标 |
|---|---|---|---|
| 1 | 显存常驻 buffer + 多 kernel 链式 | ⬜ 未开始 | 把"每次一次往返"变成"搬一次算很多步"，让 GPU 在链式 workload 上真的赢 |
| 2 | ABI 漂移检查接进 `zig build test` | ✅ `fda10dc` | 防止依赖升级悄悄改签名、只在单一目标上崩 |
| 3 | 新 kernel：reduce / GEMM | ⬜ 未开始 | 做"计算量压过搬运量"的 kernel，第一次让 GPU 明显胜出 |

建议顺序：**3（GEMM 简单版）→ 1 → 3（reduce）**。
理由：GEMM 单次就能赢，是最快的"值不值得用 GPU"的验证；但要让它成为真实工作负载的一部分，必须先有 Step 1 的常驻 + 链式；reduce 是"共享内存 + 多级归并"的低风险台阶，为 GEMM 分块优化打基础。

---

## 2. 术语速查（代码里出现过的词）

| 词 | 一句话 |
|---|---|
| kernel | 跑在显卡上的并行小函数（我们写在 WGSL 里） |
| dispatch | 派一批工人去跑这个 kernel |
| workgroup | 工人小组（当前设 64 人一组） |
| invocation | 组里的一个工人（一个线程） |
| 共享内存 / 小黑板 | 同一组工人共用的超快片上小缓存（比显存快很多） |
| staging buffer | 中转区（结果先放这里，再搬回 CPU） |
| readback | 把数据从显卡搬回 CPU |
| 往返 / roundtrip | 上传 + 回读这一次来回 |
| 常驻 | 数据放显存里不搬回来 |
| pipeline | 编译好的 kernel + 配置（套餐） |
| bind group | 告诉 kernel"用哪些 buffer"的清单 |
| uniform / params | 传小参数（如元素个数）的小块内存 |
| WGSL / SPIR-V | 显卡程序的两种格式（浏览器只吃 WGSL，见 `deps/README.md`） |
| barrier / sync | 组内工人之间的"先写完再一起读" |
| reduce | 把一大串数压成一个小结果（求和/最大值） |
| GEMM | 矩阵乘法，`A(m×k) × B(k×n) = C(m×n)` |
| GFLOP/s | 每秒浮点运算次数（算力单位） |
| GB/s | 每秒搬运字节数（带宽单位，CLI 打印的就是它） |

---

## 3. Step 1 — 显存常驻 buffer + 多 kernel 链式

### 3.1 现在的样子（每次 `add` 一次往返）

```
书桌 A,B  ──搬过去──▶  工具间：算 A+B  ──搬回来──▶  书桌 C      ← 每次都这样
```

代码位置：`src/gpu/pipeline.zig` 的 `add()` 内部就完成了"上传 → dispatch → 提交 → copy → mapAsync → readback"整条链；
`src/buffer.zig` 的 `DeviceBuffer` 目前主要是 CPU 数组 + `gpu_handle` 占位，`toDevice()/toHost()` 还是 no-op。

### 3.2 应该的样子

```
书桌 A,B,C,D ──搬一次──▶ 工具间置物架上放着 A,B,C,D
                              ├─ kernel1(A,B) → C   （C 留在显存）
                              ├─ kernel2(C,D) → E
                              └─ kernel3(E,E) → F
                                        └──搬一次──▶ 书桌 F
```

### 3.3 要落地的三件具体事

1. **`DeviceBuffer` 真正"住"在显存里**
   - 持有真实的 device buffer（不是 `u64` 占位）；
   - 记录"CPU 侧镜像 / 显存侧副本，哪边是最新的"（脏标记），避免无意义的重复上传；
   - 记 buffer usage（Storage / CopySrc / CopyDst / MapRead），当前是按需临时创建。
2. **把上传/下载从 `add()` 里挪出来**
   - 现在是"调用一次 `add` 就偷偷搬一趟"；
   - 目标：显式两步 —— `toDevice()`（先搬上去）/ `toHost()`（最后取回来）；
   - 计算函数只负责"在显存里排一个 kernel"，不负责搬运。
3. **支持一次提交多步（链式）**
   - 连续记录多个 kernel 到一个 command buffer，只提交一次、只回读一次；
   - `src/bench.zig::timeGpuAddBatched` 已经是这个方向的测量雏形，可以据此扩展成真正的 API。

### 3.4 什么时候有意义 / 没意义

- **有意义**：很多步、每步吃上一步结果的工作 —— 粒子物理、图像滤镜链、迭代求解、神经网络连续层。
- **没意义**：算一次就要结果 —— GPU 永远输给 CPU，别浪费时间。

### 3.5 验收标准

> 构造一个"链式 N 步"的 workload（建议先用 `saxpy` 或 `mul-add` 串起来），
> 让 **GPU 端到端超过 CPU SIMD**，并且**提速随链长增长**（唯一那次搬运被摊薄）。
> 报告里必须同时给：CPU SIMD 的端到端、GPU 单次往返、GPU 链式 N=1/4/16/64 的端到端。

### 3.6 风险 / 注意

- 不要为了"看起来快"去掉 `toHost()` 后的正确性校验（链式每一步都要能对拍 CPU）。
- 链式 workload 的数值要和 CPU 版逐元素对齐（浮点顺序不同会有微小差异，测试里要定容差策略，并在 README 写清楚）。
- 浏览器路径（wasm）同样受益，但浏览器侧的内存拷贝更贵，链式的收益会**更大**；改完要重跑 `zig build wasm` + 页面自证（`GPU add: MATCH`）。

---

## 4. Step 2 — ABI 漂移检查（✅ 已完成，`fda10dc`）

native（wgpu-native）与 browser（emdawnwebgpu）共用同一套 `src/gpu/webgpu.zig` 绑定，
前提是两边 `webgpu.h` 的 compute 子集签名逐字一致。依赖升级若打破这个前提，
会出现"只在某一个目标上崩溃"的诡异问题，所以把它做成测试的一部分。

| 场景 | 行为 |
|---|---|
| 正常 | `[abi] OK: 31 compute symbols identical` |
| 签名漂移 | **`zig build test` 失败**（exit 1）；单元测试本身仍全绿 |
| 版本与 `deps/pins.env` 不一致 | 打 `WARN`（升级了依赖但没重新解包的典型症状） |
| 新环境没跑过 `zig build wasm`（`.em-cache` 空） | 打 `SKIP` + 提示命令，不阻塞测试 |
| 要硬性失败（CI / 升级依赖） | `zig build abi-check`（缺输入 exit 2） |

实现：`tools/check_abi_drift.sh` + `build.zig` 里 `test` step 依赖它、并新增 `abi-check` step。

---

## 5. Step 3 — 新 kernel：reduce / GEMM

### 5.1 为什么 `add` 这类没前途

`add` 是"读 2 个数、写 1 个数"，**每个数据只被用一次**。这种活儿的时间几乎全花在搬运上，
GPU 的几十个工人根本没发挥空间 —— 上面的实测数字（3.5 GB/s vs 13 GB/s）就是证明。

### 5.2 reduce（归约）

**是什么**：把一大串数压成一个小结果 —— 求和、求最大值、求平均。数据多、结果少。

**为什么有价值**：
- 读 100 万个数、只写出 1 个数；
- 加法可以"两两归并"（100万 → 50万 → … → 1），适合让一组工人先把各自那段加起来，再逐级归并；
- 实际用处：统计、归一化、softmax —— 做 AI/图像/物理几乎绕不开。

**难点**：多级归并的次数与线程规模要匹配；尾部不满一组要单独处理；组间归并需要第二个 kernel 或原子操作。

**验收标准**：sum / max 与 CPU 结果一致（浮点求和需定容差），并在大数组上报出不输于 `cpu_simd` 的时间。

### 5.3 GEMM（GPU 主场）

**是什么**：`A(m×k) × B(k×n) = C(m×n)`，两个二维数表相乘。

**为什么 GPU 碾压**：
- 算 `m×n` 的输出，每个元素要做 `k` 次乘加 → 总共约 `2·m·n·k` 次运算；
- 需要搬的数据只有 `m·k + k·n + m·n` 个；
- **同一块数据被重复使用很多次**（`B` 的每一列会被 `A` 的所有行反复用）；
- 于是可以先把 `A`、`B` 的一小方块抄到**共享内存（小黑板）**上，一组工人反复使用它算出结果，再换下一块。
- 一句话：**搬运只做一次，计算做很多次** → 计算量越大，GPU 优势越大。
- 经验值：CPU 通常几十到上百 GFLOP/s，笔记本 GPU 轻松上千 GFLOP/s（差几十倍）。

**为什么比 `add` 难**：
1. **分块（tiling）**：大矩阵要切成能塞进共享内存的小块；
2. **同步**：同组工人必须"先写完小黑板、再一起读"（顺序错了结果就错）；
3. **边界**：行列不是块大小整数倍时要特殊处理；
4. **数据布局**（行主序/列主序、是否转置）能让性能差好几倍。

**两步走（强烈建议）**：
1. **先写"慢但正确"的版本**：每个工人算一个输出元素，只用显存、不碰共享内存；
   用它对拍 CPU（含非整数倍边界、`k=1`、`m/n/k` 很小时的特例）。
2. **再做共享内存分块优化**：同一套测试必须继续全绿，报告分块前后 GFLOP/s。

**验收标准**：
- 正确性：与 CPU 参考实现逐元素一致（容差策略写进 README）；
- 性能：在 `m=n=k=512` 和 `1024` 上报告 GFLOP/s，并与 `cpu_simd` 版对比；
- 明确写出"简单版 vs 分块版"的差距，证明优化有效。

### 5.4 与 Step 1 的关系

- 单次 GEMM 的算力密度高，**不依赖 Step 1 就能赢**；
- 但真实工作负载是"连续多步"（例如 `A×B×C` 或前向+反向），此时必须靠 Step 1 的常驻 + 链式才不浪费；
- 所以两者是互补：**Step 3 证明"值得用 GPU"，Step 1 让"用起来不浪费"**。

---

## 6. 任务拆分建议（可派 sub-agent）

| 任务 | 内容 | 依赖 | 建议规模 |
|---|---|---|---|
| T5 | GEMM 简单版（全局内存）+ CPU 参考对拍 + GFLOP/s 基准 | 无 | 中 |
| T6 | GEMM 共享内存分块版 + 性能对比 | T5 | 中 |
| T7 | reduce（sum/max）+ 归并 + 基准 | 无（可与 T5 并行，只读同一份绑定） | 小 |
| T8 | 显存常驻 buffer + 显式 `toDevice/toHost` + 链式提交 API | T5 或 T7（要有第二个 kernel 才能真正体现链式） | 大 |

派单时的固定约定（沿用前面几个任务的经验）：
- 交接文档写进 `handoff.md`；`handoff-3steps.md`（本文）只做路线图，不改。
- 约束：不用 `wgpuInstanceWaitAny` / `wgpuDevicePoll` / SPIR-V / ASYNCIFY / push constants(immediates)。
- 不许 kill 任何进程；长任务用 `nohup setsid pi ...` 后台跑 + 轮询（阻塞调用会被工具超时连带杀掉）。
- 每个任务跑通后立刻 `git commit`，commit message 里带实测数字。

## 7. 命令速查

```bash
zig build test --summary all                 # 15/15 + [abi] 检查
zig build abi-check                          # 严格 ABI 检查
zig build run -- --backend gpu_webgpu --size 8388608 --iters 5
zig build run -- --auto --size 1048576       # 诚实选择（当前选 cpu_simd）
zig build wasm && (cd zig-out/webgpu && python3 -m http.server 8090)
tools/check_abi_drift.sh                     # 单独跑 ABI 检查
```
