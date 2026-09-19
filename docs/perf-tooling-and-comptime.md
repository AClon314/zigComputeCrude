# 性能分析与 comptime 判据（2026-09-19）

问题：本仓库的 Zig 代码里，**哪些改成 comptime 能明显提升性能**？
做法：先用二进制级热力工具定位热路径，再对每个候选做同机 A/B 实测，只留有效果的。

---

## 0. 结论

**CPU 侧只有两处 comptime 化有实质收益，都已落地**（§3.1 / §3.2）：

| 旋钮 | 位置 | 实测提升 |
|---|---|---|
| GEMM 寄存器分块行数 `cpu_block_rows` | `src/gpu/gemm.zig` | 256³ 46.5 → 71.4、512³ 33.7 → 58.9、1024³ 18.0 → 42.6 GFLOP/s（1.5~2.4x） |
| SIMD 归约累加器数 `acc_vectors` | `src/gpu/reduce.zig` | sum 16 MiB 22.9 → 29.6 GB/s；**256 KiB（L2 内）43.3 → 94.1 GB/s（2.2x）**；max 256 KiB 52.7 → 85.2（1.6x） |

**另外两处实测无收益，因此没有改**（§3.3 / §3.4）：elementwise 循环展开、WGSL 里把 kNN 容量
`K` 变成编译期常量。

**指令热力显示没有"运行时分派热点"可提升**：`add` 的热力里 91% 指令都在 SIMD 循环本体
（§2.1），没有 per-element switch、没有虚调用、没有可 hoist 的重复工作。所以本仓库的
comptime 机会不在"消灭运行时分派"，而在**把旋钮从常量变成编译期常量，好让 LLVM 静态展开
累加器/寄存器分块**。

---

## 1. 工具：二进制热力图怎么来

| 工具 | 装法 | 用途 | 本机状态 |
|---|---|---|---|
| `valgrind --tool=callgrind` + `callgrind_annotate` | `pixi global install valgrind`（conda-forge，无需特权） | **指令级热力**：每个函数的 Ir 占比、调用图、逐行标注 | ✅ 已装 |
| `valgrind --tool=cachegrind` + `cg_annotate --show=Ir,D1mr,DLmr` | 同上 | **缓存热力**：D1/LL 失配数（判断是否 bandwidth-bound 的量化手段） | ✅ 已装 |
| `samply`（采样+火焰图） | `cargo install samply --locked` | 真·时间采样火焰图，输出可用 profiler.firefox.com 打开 | ⚠️ 已装 v0.13.1，但本机 `perf_event_paranoid=2`，需 `sudo sysctl kernel.perf_event_paranoid=1` |
| `perf` + `hotspot`（GUI） | `sudo dnf install perf hotspot` | 系统级采样 + 友好的热力图界面（KDE/Qt） | ❌ 未装（需要 sudo，按仓库约定由用户安装） |
| 本仓库 CLI 计时 | `zig build -Doptimize=ReleaseFast` | **最终裁决**：真实端到端、ReleaseFast、跑 3 次取中位数 | ✅ |

```bash
# 指令热力（CPU-only 负载最干净；GPU 初始化会污染小负载的 profile，见 §2.1）
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out ./zig-out/bin/computeAccel \
    --backend cpu_simd --kernel add --size 1048576 --iters 20
callgrind_annotate --threshold=90 --auto=yes /tmp/cg.out | head -30

# 缓存热力
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=/tmp/cg_cache.out \
    ./zig-out/bin/computeAccel --backend cpu_simd --kernel add --size 4194304 --iters 3
cg_annotate --show=Ir,D1mr,DLmr /tmp/cg_cache.out | head -20

# 时间采样火焰图（需要先生效 paranoid=1）
sudo sysctl kernel.perf_event_paranoid=1
samply record --save-only --output /tmp/prof.json.gz -- \
    ./zig-out/bin/computeAccel --kernel gemm --m 1024 --k 1024 --n 1024 --variant simple --iters 20
# 然后打开 https://profiler.firefox.com 载入 /tmp/prof.json.gz
```

**注意**：`valgrind` 记录的是**指令数/访存次数**，不是时间；对"带宽受限"的判断必须配合
cachegrind 的失配数据和 CLI 的真实计时，不能只看 Ir 占比。

---

## 2. 实测热力

### 2.1 指令热力

`--backend cpu_simd --kernel add --size 1048576 --iters 20`（CPU-only，无 GPU 初始化）：

```
Ir                   file:function
71,221,361 (86.93%)  src/engine.zig:main.runAddDemo      <- addScalar/addSimd 被内联进来
 8,847,548 (10.80%)  src/engine.zig:bench.timeAdd__anon_3439
 1,048,801 ( 1.28%)  src/main.zig:main.runAddDemo
```

即 **97.7% 的指令在 SIMD 循环本体 + 计时循环里**，其余（参数解析、打印、分配）都是零头。
没有运行期算子分派、没有重复计算可以 hoist ——"改成 comptime 省掉分派"这一类优化在本仓库
**没有目标**。

⚠️ 反例（方法学坑）：`--kernel gemm --m 128` 的 profile 里前几名是
`libnvidia-eglcore` 14.7% / `libnvidia-glcore` 14.0% / `ld.so` 5.7% + Vulkan loader + expat
—— 那是**进程启动 + GPU 驱动初始化**，把 128³ 的 kernel 时间淹没了。分析 CPU 侧要用
CPU-only 负载（`--backend cpu_simd`）或足够大的 workload。

### 2.2 缓存热力（为什么 elementwise 没法再快）

`add` 16 MiB × 3 条流（读 a、读 b、写 out），iters=3：

```
Ir 54.55M   D1mr 3,679,057   DLmr 3,677,578     (D1 失配的 99.97% 直达最后一级)
engine.zig runAddDemo: Ir 81.1%, D1mr 57.0%, DLmr 57.0%
```

D1 失配几乎全部直达内存 ⇒ 纯 **compulsory（流式）失配**，没有可复用的局部性。
对这样的负载，循环展开/预取/SIMD 宽度之类"指令侧"手段都只是换个方式等内存：

- 实测 elementwise 展开 4 向量一次：1 MiB 56.1 → 52.6、16 MiB 10.55 → 10.56、
  64 MiB 9.26 → 9.53 GB/s（全在噪声内，1 MiB 甚至更差）。
- 相反，**有依赖链**的内核（归约）展开累加器能立刻见效：见 §3.2 的 2.2x。

这把"该改哪个"讲清楚了：**先看这个内核是等内存还是等依赖链**。

---

## 3. comptime 候选：A/B 实测与判定

测量口径：`zig build -Doptimize=ReleaseFast`，CLI 真实端到端，3 次取中位数，
机器非独占（Ryzen 5 5600H + Vega iGPU + RTX 3050），下同。

### 3.1 ✅ GEMM 寄存器分块行数（`cpu_block_rows`，comptime）

`referenceSimd` 内层是 i-k-j：每读 `width` 个 B 元素服务 ROWS 个输出行，B 的读取总量是
`(m/ROWS)·k·n`，ROWS 直接决定"每字节 B 换来几次 FMA"。原来硬编码 4，改成 comptime 参数后：

| ROWS | 256³ | 512³ | 1024³ |
|---|---|---|---|
| 4（原值） | 46.5 | 33.7 | 18.0 |
| 6 | 61.8 | 51.6 | 32.8 |
| **8（新默认）** | **71.4** | **58.9** | **42.6** |
| 10 | 77.8 | 61.6 | 44.2 |

- 8~10 是平台期；12 在 1024³ 回落（寄存器溢出，prototype 实测 39.5）→ 取 8。
- **不改变舍入顺序**：每个输出元素仍是 k 升序 `@mulAdd`，ROWS 只决定"同时算几行"，
  因此新旧结果逐位相同；新增测试 `referenceSimdBlocked is bit-identical across register
  block sizes` 把这条不变量钉住（ROWS ∈ {1,2,3,8,16} 全部逐位相等）。
- `noalias` 实测无差别（±1%），因此没加：那会引入"输出不得与输入别名"的隐性契约。

### 3.2 ✅ SIMD 归约累加器数（`acc_vectors`，comptime）

单累加器时 `lanes += values` 是一条长度 = len/width 的依赖链，被加法延迟卡住；
拆成 ACC 条链就能吃满加载带宽。CLI `--kernel reduce`（cpu_simd GB/s，3 次中位数）：

| ACC | sum 16 MiB | sum 256 KiB | max 16 MiB | max 256 KiB |
|---|---|---|---|---|
| 1（原值） | 22.9 | 43.3 | 24.5 | 52.7 |
| 2 | 29.6 | 78.1 | 25.8 | 83.5 |
| **4（新默认）** | **29.6** | **93.1** | **26.3** | **83.5** |
| 8 | 28.2 | 93.1 | 26.7 | 82.0 |

- 16 MiB 是内存带宽受限（+29%），**256 KiB（L2 内）原本是延迟受限 → 2.2x**。这条最能说明
  "先判断瓶颈类型"：同一段代码，尺寸不同、瓶颈不同、旋钮收益不同。
- `max` 在 ACC=2 就饱和；统一取 4 对 sum 最好、对 max 无损失。
- 代价：归约的**结合顺序改变**（文档里本来就写明"与标量参考不同、按容差比"），
  所有相关测试（1e-5 / 1e-4）与 GPU 对拍都通过。

### 3.3 ❌ elementwise 循环展开（不做）

`add`/`saxpy` 的 UNROLL=4 版本 vs 原版（GB/s，5 次中位数）：

| 尺寸 | 原版 | UNROLL=4 |
|---|---|---|
| 1 MiB | 56.1 | 52.6 |
| 16 MiB | 10.55 | 10.56 |
| 64 MiB | 9.26 | 9.53 |

全在噪声内（1 MiB 甚至更差）。原因见 §2.2（纯流式失配）。**未改**，也建议以后不要加。

### 3.4 ❌ 把 kNN 容量 K 变成 WGSL 编译期常量（不做）

`grid_query.wgsl` 目前从 uniform 读 `params.max_neighbors`。把它换成 shader 里的
`const K: u32 = 16u`（即"shader 特化"），在本机实测（`--kernel spatial --points 262144
--queries 4096`，`gpu_query` 稳态中位数）：

| 环境 | uniform（现状） | 编译期常量 |
|---|---|---|
| iGPU | 2.642 ms | 2.791 ms |
| dGPU | 0.606 ms | 0.615 ms |

**无收益**（差异在 ±5% 噪声内）：这个 kernel 的成本在"遍历 cell + 随机读 slot"的内存延迟，
`total < K` 的比较与 `q*K` 的乘法本来就是零头。所以**不引入**"override 常量 + 按 K 特化
pipeline 缓存"那套机制（成本不小，收益为零）。

---

## 4. 只能留到运行时的东西（comptime 不是万能的）

| 东西 | 为什么不能 comptime | 现状 |
|---|---|---|
| 形状/规模 `m,k,n,len,groups` | 运行期数据（调用方给） | 已 comptime 化的是**后端/向量宽度/算子类**，不是尺寸；尺寸只用来算 grid |
| adapter 偏好、`--precision` 档、CLI flags | 策略/外部输入 | 策略是运行期开关，但**判据表**本身是 comptime（`determinism.zig`） |
| device limits / 能力探测 | 运行期查询 | `probe()` + limits 闸门 |
| WGSL 源码与 entry point | shader 是运行期字符串交给驱动 | 除"override 常量"（§3.4 实测不值得）外没有 comptime 空间 |
| `VariantCache`/buffer 的 shape key | 运行期数据 | 每调用一次的哈希 ≪ kernel 本身，不值得 |
| 缓存大小、tile 尺寸的**自动**选择 | 需要运行期量测 | 只把"分块行数/累加器数"这类**与尺寸无关**的旋钮做成 comptime |
| 进程/线程级并行 | 属于调度，不是 comptime | 当前 CLI 单线程；这是下一个数量级的机会（未做） |

---

## 5. 复现命令

```bash
zig build -Doptimize=ReleaseFast

# §3.1 GEMM 旋钮（把 src/gpu/gemm.zig 的 cpu_block_rows 改成 4/6/8/10 各跑一遍）
for n in 256 512 1024; do for r in 1 2 3; do
  ./zig-out/bin/computeAccel --kernel gemm --m $n --k $n --n $n --variant simple --iters 5 \
    | awk '/^cpu_simd/{print "'$n'³", $3}'; done; done

# §3.2 归约旋钮（同理改 src/gpu/reduce.zig 的 acc_vectors）
for n in 16777216 262144; do ./zig-out/bin/computeAccel --kernel reduce --size $n --op sum --iters 10; done

# §3.3 elementwise（UNROLL 需要自己临时改 engine.zig 对比）
./zig-out/bin/computeAccel --backend cpu_simd --kernel add --size 16777216 --iters 10

# §3.4 K 特化（临时把 grid_query.wgsl 的 params.max_neighbors 换成 const）
./zig-out/bin/computeAccel --kernel spatial --points 262144 --queries 4096 --radius 2.0 --iters 3 --adapter high-perf
```

⚠️ 机器非独占：同一二进制同一负载的 run-to-run 波动可见 ±5%（小尺寸 L2 负载可达 ±20%），
所以结论只认"多次中位数 + 跨尺寸一致"的信号，不认单次最优值。
