# PEFT：LoftQ 量化基座与交替 SVD 初始化

## 题目描述

**LoftQ**（LoRA-Fine-Tuning-aware Quantization，arXiv:2310.08602）在量化基座权重的同时，用交替 SVD 初始化 LoRA 的 \(A/B\)，使得：

\[W_0 \approx W_q + B A,\]

其中 \(W_q\) 是量化后的基座。交替流程如下：

1. 对 \(W_0\) 做 \(b\)-bit 对称 absmax 量化，得到 \(W_q^{(0)}\)。
2. 对迭代 \(t=0,\dots,\text{iterations}-1\)：
   - 残差 \(R = W_0 - W_q^{(t)}\)。
   - 对 \(R\) 做 SVD：\(R = U S V^\top\)。
   - 取前 \(r\) 个奇异值：\(s_r = \sqrt{S_{:r}}\)。
   - 初始化 \(B = U_{:, :r} \cdot s_r\)，\(A = s_r \cdot (V^\top)_{:r, :}\)。
   - 更新量化基座 \(W_q^{(t+1)} = \operatorname{quantize}_b(W_0 - B A)\)。
3. 返回合并权重 \(W' = W_q^{(T)} + (\alpha/r)\,B A\)。

当 `iterations=0` 时，\(B,A\) 为零矩阵，\(W'\) 直接等于 \(W_0\) 的 \(b\)-bit 量化结果。

## 任务

给定 \(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)、秩 \(r\)、缩放 \(\alpha\)、量化位数 `bits`（正整数）、交替轮数 `iterations`（非负整数），按上述 LoftQ 流程返回合并权重 \(W'\)。

**量化函数**定义为对称 absmax 量化：

\[\operatorname{quantize}_b(W) = \mathrm{round}\!\left(\frac{W}{s}\cdot q_\max\right)\cdot \frac{s}{q_\max},\qquad s=\max|W|,\ q_\max=2^{b-1}-1.\]

当 \(s=0\) 时直接返回 \(W\)（全零矩阵无需量化）。

- `iterations=0` 退化为纯量化基座。
- \(\alpha=0\) 时同样只返回最终量化基座 \(W_q\)。
- 输出 \(W'\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，float32。

## 提示

- 使用 `torch.linalg.svd(R, full_matrices=False)` 得到 \(U,S,Vh\)（其中 \(Vh=V^\top\)）。
- \(B A\) 恰好等于 \(U_{:, :r}\,\operatorname{diag}(S_{:r})\,(V^\top)_{:r, :}\)，也就是残差 \(R\) 的最佳秩-\(r\) 逼近；因此只判断最终合并权重 \(W'\) 即可，不判断 \(U/V\) 的符号。
- 对 \(S_{:r}\) 取平方根后分别乘到 \(U\) 和 \(Vh\) 上，得到平衡的 \(B/A\) 初始化。
