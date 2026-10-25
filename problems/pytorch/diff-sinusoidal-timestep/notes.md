# 详解：正弦时间步嵌入

## 出处
Ho et al., *Denoising Diffusion Probabilistic Models*（arXiv:2006.11239）附录对 \(t\) 采用与 Vaswani Transformer 位置编码同源的固定正弦特征；openai/improved-diffusion 的 `timestep_embedding` 即此实现。用一组跨越多个数量级的频率把标量 \(t\) 展开成高维、平滑、可微的表示。

## 核心与易错点
1. **half 与频率指数**：\(\text{freq}_i=10000^{-i/\text{half}}\)，指数分母是 `half`（等价 \(2i/\text{dim}\)）。用错分母会整体压频。
2. **拼接顺序**：本题口径是「前 half 个全 sin、后 half 个全 cos」（`cat([sin,cos])`），不是 sin/cos 交错。顺序不同则向量逐元素不等——务必与题面一致。
3. **\(t=0\) 探针**：\(\sin 0=0\)、\(\cos 0=1\)，可一眼检查 sin/cos 是否放错半段。
4. float32：`arange` 要显式 `dtype=torch.float32`，否则默认整数会报错。

## 对拍口径
参考解向量化：`freqs=exp(-log(10000)*arange(half)/half)`、`cat([sin(t*freqs),cos(t*freqs)])`。判题 `_oracle` 用 Python `math` 逐 i 计算 `angle=t/10000**(i/half)` 分别填 sin 半段与 cos 半段（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设 `t=0` 边界与不同 dim 的形状/一致性检查。

## 复杂度
\(O(\text{dim})\)，逐频率一次三角函数。
