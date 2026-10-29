# 解题说明 · Min-SNR 损失加权

## 出处

- `huggingface/diffusers` 的 `src/diffusers/trainers/training_utils.py`：`compute_snr` 从 `alphas_cumprod` 取 `alpha = sqrt(ᾱ)[t]`、`sigma = sqrt(1-ᾱ)[t]`，再算 `snr = (alpha / sigma) ** 2`；源码注释注明该式取自 `TiankaiHang/Min-SNR-Diffusion-Training` 的 `guided_diffusion/gaussian_diffusion.py`。
- Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*（arXiv:2303.09556）给出 \(\min(\mathrm{SNR},\gamma)/\mathrm{SNR}\) 的加权形式，常用 \(\gamma=5\)；训练脚本把它乘在每步的 MSE 上。
- 本题只做「调度 + 时间步 → SNR 与权重」这一层纯计算，不涉及损失、反传或噪声采样。

## 核心与易错点

**被截断的是 SNR，不是权重。** 写 `w = min(snr, gamma)` 会得到大于 1 的权重（SNR 可以到几千），正确写法是 `min(snr, gamma) / snr`，等价于 `min(1, gamma / snr)`。两种写法在 `snr <= gamma` 时都等于 1，错误只在干净端暴露——测试专门用高 SNR 的时间步抓这一点。

**SNR 走平方根路径，不是直接 `ᾱ/(1-ᾱ)`。** 源码先算 `sqrt(ᾱ)` 与 `sqrt(1-ᾱ)` 再相除平方；数学上等价，但 float32 的舍入不同。题目按源码路径计算，并用 `ᾱ/(1-ᾱ)` 作为独立 oracle，容差放到 1e-5。

**`alphas_cumprod` 的索引方向。** 下标 0 对应最干净（\(\bar\alpha\) 最大、SNR 最大），下标 \(T-1\) 最噪声；越界检查必须用 `max(t) < T`，负下标在 PyTorch 里会被当成倒数索引，若不拦截会静默算出别的值。

**严格开区间 \((0,1)\)。** \(\bar\alpha=1\) 时 \(\sigma=0\)，SNR 发散；\(\bar\alpha=0\) 时 \(\alpha=0\)，权重变成 \(0/0\)。源调度通常不会取到端点，但契约上必须显式拦住，否则随机测试会偶发 inf/nan。

**`gamma` 是 SNR 上限。** \(\gamma\) 越小，越多时间步被封顶、权重越低；\(\gamma\to\infty\) 时所有权重为 1（退化为普通 MSE）。测试用单调性把方向钉死：\(\gamma\) 变大，权重逐点不减。

**输出精度与形状。** 内部 float64、出口 float32；形状跟随 `timesteps`（不是 `alphas_cumprod`），重复下标必须给出完全相同的结果，空张量合法。

## 与其他题的分工

- `diff-zero-terminal-snr`：改的是 \(\bar\alpha_t\) 的端点，让最噪声端 SNR 归零；本题不改调度，只读调度算权重。
- `diff-sd3-timestep-density`：抽时间步的密度；本题是抽完之后给损失加权，二者拼成训练侧的时间步管线。
- `to-*` 系列（DPO/IPO 等）：语言模型的 token 级目标；本题是扩散模型的逐步 MSE 加权，维度与语义都不同。
- `diff-ddpm-forward` 里的后验均值/方差用到同一份 `alphas_cumprod`，但那是前向加噪账；本题只用到 SNR。
