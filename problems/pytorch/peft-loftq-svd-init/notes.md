# PEFT：LoftQ 量化基座与交替 SVD 初始化——算法详解

## 1. 公式与出处

LoftQ 出自 arXiv:2310.08602，目标是把全精度基座 \(W_0\) 同时近似成「低 bit 量化权重 + 低秩适配器」之和：

\[W_0 \approx W_q + BA.\]

交替算法的每一步都用当前 \(W_q\) 的残差做最佳秩-\(r\) 逼近（SVD），再用该逼近更新 \(W_q\) 的量化目标，反复收紧。

## 2. 对称 absmax 量化

本题采用简化的 per-tensor 对称 absmax 量化：

\[\operatorname{quantize}_b(W)=\mathrm{round}\!\left(\frac{W}{s}\cdot q_\max\right)\cdot \frac{s}{q_\max},\quad s=\max|W|,\ q_\max=2^{b-1}-1.\]

- 当 \(s=0\) 时直接返回 \(W\)。
- 只考量化 + SVD 初始化流程，不涉及 NF4 等复杂码本；最终合并权重与具体量化网格细节无关。

## 3. SVD 初始化细节

- 残差 \(R = W_0 - W_q\)。
- `torch.linalg.svd(R, full_matrices=False)` 返回 \(U,S,Vh=V^\top\)。
- \(s_r = \sqrt{S_{:r}}\)，\(B = U_{:, :r} \cdot s_r\)，\(A = s_r \cdot Vh_{:r, :}\)。
- 因此 \(BA = U_{:, :r}\,\operatorname{diag}(S_{:r})\,Vh_{:r, :}\)，即 \(R\) 的最佳秩-\(r\) 逼近。

## 4. 退化与判题考点

- `iterations=0`：\(B,A\) 为零，\(W'\) 等于 \(W_0\) 的量化结果。
- \(\alpha=0\)：LoRA 项为零，\(W'\) 等于最终量化基座 \(W_q\)。
- 全零 \(W_0\)：scale 为零，量化恒等，结果仍为零。
- 判题只断言最终合并权重 \(W'\)，因为 SVD 的奇异向量符号/简并序不确定，直接断言会翻车。

## 5. 与现有题的分工

- `peft-qlora-nf4-forward`：NF4 量化后的前向（含反量化矩阵乘）。
- `quant-double-quant-scale`：scale 张量的再量化。
- 本题：LoftQ 特有的「量化基座 + 交替 SVD 初始化 LoRA」的合并结果。
