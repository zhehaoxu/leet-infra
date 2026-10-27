# PEFT：QLoRA NF4 反量化前向与显存账——算法详解

## 1. NF4 码本与分位数构造

QLoRA（Dettmers et al., NeurIPS 2023, arXiv:2305.14314）的 NF4 是"固定 16 值码本 + 块缩放"的查表式量化：16 个重建值取标准正态分布落在 \([-1,1]\) 内、概率质量均分（每段 1/16）的分位点，因此对正态分布的权重信息论最优（论文与 int4/int8 均匀格点的对比消融即为此设计）。本合集与 #85①（MX FP8/E8M0 微缩放）分工：MX 是"逐元素组共享浮点指数"的浮点格式，NF4 是固定码本查表，两者块结构（64）同源但机制不同。

码本的三个锚点：码 **7** 恰为 \(0.0\)（正态对称中点），码 **0/15** 为 \(\pm1\)（饱和边界）——`check_nf4_code_values` 用它们做独立锚点断言。

## 2. 块缩放与 double quantization（判题考点）

- 基础块量化：每 64 个权重（同一行的连续列）一个 FP32 absmax，开销 \(32/64=0.5\) bit/param。
- **double quantization**：每个 block 的 FP32 absmax 量化为 FP8（1 字节），每 256 个 block 共享一个 FP32 二级缩放：

\[8/64+32/(64\cdot 256)=0.125+0.001953\approx 0.127\ \text{bit/param}\]

  比 0.5 省 0.373 bit/param——论文以此把 65B 模型的缩放开销从 ~4GB 量级压到 ~1GB 量级。
- 本题字节数（\(n\) 为权重个数，\(d_\text{in}\) 是 64 的倍数）：
  - 权重：\(n/2\)（4-bit 打包，\(n\) 为偶数）
  - 一级缩放：\(n/64\) 个 FP8 → \(n/64\) 字节
  - 二级缩放：\(\lceil (n/64)/256\rceil\) 个 FP32 → \(\times 4\) 字节

`check_bytes_account` 对 \((1,64)\)、\((2,128)\)、\((4,640)\) 三个尺寸断言 \(37/136/1324\)。

## 3. 前向组合（判题考点）

QLoRA 前向 = 反量化基座（冻结）+ BF16 LoRA 旁路（可训）：

\[y=xW^{\!\top}+\frac{\alpha}{r}(xA^{\!\top})B^{\!\top}\]

工程实现（bitsandbytes `Linear4bit`）里反量化在 kernel 内即时进行、从不物化整个 \(W\)——本题为了判题确定性把 \(W\) 物化出来。`check_lora_branch_scales` 断言旁路输出随 \(\alpha\) 严格线性：

\[y(2\alpha)-y(0)=2\,\bigl(y(\alpha)-y(0)\bigr)\]

把"量化基座冻结、旁路加性"的语义钉死；两条路径任何一条算错（比如把缩放乘到基座上、或把 \(\alpha/r\) 记成 \(\alpha\))都会破坏这个等式。

## 4. 与本合集其他题的分工

- `peft-lora-forward`/`peft-lora-merge`：非量化基座上的 LoRA 前向与合并。
- `#85①`（MX/E8M0）：另一种量化格式的缩放原语，勿与 NF4 码本互抄。
- 本题：QLoRA 的"反量化贯通 + double-quant 显存账"两件事；量化（编码）方向、pivot vector 反传、paged optimizer 均不在范围。

## 5. 验证方法

- `check_example`：\(1\times64\) 单 block、absmax \(=2\)、全 7 底（码值 0.0）+ 三个特殊码位（0/8/15）→ \(y=0.159\)、nbytes \(=37\)。
- `check_matches_oracle`：随机 \(4\times128\)、\(N=3\) 与逐标量独立重算对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
- 码值锚点、字节账、旁路线性性分别用固定断言。
