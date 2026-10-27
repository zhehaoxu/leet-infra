# PEFT：VeRA 共享冻结基矩阵与可学缩放——算法详解

## 1. 出处与核心思想

Kopiczko et al.（ICLR 2024, arXiv:2310.11454）。LoRA 的参数 \(r(d_\text{out}+d_\text{in})\) 里，真正"学到的方向"高度冗余——VeRA 把 \(A,B\) 变成**跨层共享的随机冻结基**（一次生成、所有层复用），每层只学两个缩放向量：

- \(\lambda_d\in\mathbb R^r\)：作用在低秩**隐藏维**上（缩放 \(xA^\top\) 的 r 个分量）；
- \(\lambda_b\in\mathbb R^{d_\text{out}}\)：作用在**输出维**上（缩放投影回 d_out 后的各分量）。

等效于给共享基做"每层自己的对角重加权"。论文在 LLaMA/GPT-2 上用每层 \(d_\text{out}+r\) 个参数（约为 LoRA 的百分之一量级）拿到接近的微调质量。

## 2. 前向的两条等价路径（判题考点）

**函数式**（PEFT forward 原式）：

\[y=xW_0^{\!\top}+\lambda_b\odot\Bigl(\bigl(\lambda_d\odot(xA^{\!\top})\bigr)B^{\!\top}\Bigr)\]

**矩阵式**（merge 后）：

\[\Delta W=(\text{diag}(\lambda_b)\,B)\,(\text{diag}(\lambda_d)\,A),\qquad y=x\,(W_0+\Delta W)^{\!\top}\]

`check_matches_oracle` 的 oracle 走矩阵式、参考解走函数式，对拍钉死等价性（"只断言重建后的 ΔW/合并权重"红线在 VeRA 上的形态：无论怎么算，\(\Delta W\) 必须是同一个）。注意 \(\lambda_d\) 乘在 \((N,r)\) 的**列**上、\(\lambda_b\) 乘在 \((N,d_\text{out})\) 的**列**上——两个广播方向都别搞反。

## 3. 初始化与参数账

- PEFT `reset_vera_parameters`：\(\lambda_b\) 全零、\(\lambda_d=d_\text{initial}=0.1\)。起点处 \(\Delta W=0\)，适配器"什么都不做"——`check_zero_lambda_b_identity` 验它。
- 每层可训参数 \(d_\text{out}+r\)，对比 LoRA 的 \(r(d_\text{out}+d_\text{in})\)：`check_param_account` 对随机形状断言。
- `check_lambda_b_per_output`：\(\lambda_b\) 某位翻倍，只有对应**输出列**变化、且变化量恰为该列原适配量——逐输出门控语义。

## 4. 与本合集其他题的分工

- `peft-lora-forward`/`peft-lora-merge`：每层独立 A/B 的 LoRA。
- `peft-dora`：幅值-方向分解（每层独立因子 + 幅值向量）。
- 本题：跨层共享基 + 每层缩放向量的极端省参数形态。

## 5. 验证方法

`check_example`（\(d_\text{in}=2,d_\text{out}=2,r=1\)）：\(A=[1,1]\)、\(B=[1,-1]^{\!\top}\)、\(\lambda_d=2\)、\(\lambda_b=[1,-1]\)、\(x=[1,2]\) → 隐藏 \(=3\cdot2=6\)、适配 \([6,-6]\)、乘 \(\lambda_b\) 得 \([6,6]\)、加基座 \([1,2]\) → \(y=[7,8]\)、\(n=3\)；`check_matches_oracle` 双路径对拍（float64 内部、float32 输出、容差 \(10^{-5}\)）。
