# 详解：Adafactor 因子分解二阶矩

## 出处
Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, arXiv:1804.04235（2018）。算法骨架对齐 kozistr/pytorch_optimizer 的 `AdaFactor`（BigVision 变体、eager per-param 路径、`ams_bound=False`），并采用其行归一化外积重构 \(\hat v\)（与 HuggingFace/FairSeq Adafactor 的 `_approximate_second_moment` 同型）。

## 三个省显存/免调 LR 的点子
1. **因子化二阶矩**：矩阵 \(v\in\mathbb R^{r\times c}\) 用行统计 \(v_r\in\mathbb R^{r}\)、列统计 \(v_c\in\mathbb R^{c}\) 的外积近似，状态从 \(rc\) 降到 \(r+c\)。重构写法
   \[ \hat v_{ij}=\frac{v_{r,i}}{\sum_k v_{r,k}}\,v_{c,j} \]
   即先把行向量归一化成"每行占比"，再乘列统计。这一归一化让 \(\hat v\) 的量级回到 \(v_c\) 尺度。
2. **RMS 更新裁剪**：把预条件方向 \(u=g/\sqrt{\hat v}\) 的整体 RMS 除以 \(\text{clip\_threshold}\) 并 clamp 到 \(\ge1\) 再回除——等于限制每步更新的 RMS 不超过阈值。这是"免学习率"的来源。
3. **按参数 RMS 缩放步长**（`scale_parameter`）：\(lr_{eff}=\max(\epsilon_2,\operatorname{RMS}(p))\cdot lr\)，让大参数矩阵走更大步。

## 本题为 CPU 确定性做的约定（务必对齐题面）
- **不启用 `relative_step`**：直接用给定 `lr`，避免 \(1/\sqrt t\) 依赖步数（题面已声明）。
- **动量用 float32**，不用参考库默认的 bfloat16——否则跨后端落点不可控。
- 二阶衰减 \(\beta_{2,t}=\min(\beta_2^{cap},1-t^{-0.8})\) 里 decay_rate 默认 \(-0.8\)；EMA 权重 \(b=1-\beta_{2,t}\)，即 `v = (1-b)*v + b*U`，等价标准的 \(v\leftarrow\beta_{2,t}v+(1-\beta_{2,t})U\)。
- 一阶动量在**裁剪并乘 \(lr_{eff}\) 之后**再 EMA，最终位移用的是 \(m\) 本身（`p -= m`），不再另乘 lr。
- `eps1=1e-30` 加在 \(g^2\) 上（等价参考实现把 eps1 提前加进梯方，而非后面 clamp）。

## 易错点
- \(\hat v\) 的重构分母是 \(\sum_k v_{r,k}\)（对**行统计**求和），不是列。
- 更新裁剪的 RMS 是在 \(u=g/\sqrt{\hat v}\)（预条件后、乘 lr 前）上算的。
- 一阶动量、权重衰减、更新的顺序按题面 5→9 步。

## 对拍口径
外部参照用等价的向量化重算（显式做行/列 EMA、外积重构、RMS 裁剪、动量）。三步示例值实测 solution 与参照 maxdiff 0（float32、同算子序）。`grad is None` 跳过。
