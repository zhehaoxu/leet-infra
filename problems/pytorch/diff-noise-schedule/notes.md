# 详解：线性 / 余弦噪声调度 ᾱ 序列

## 出处
- linear：Ho et al., DDPM（arXiv:2006.11239），`betas = linspace(1e-4, 0.02, T)`。
- cosine：Nichol & Dhariwal, *Improved DDPM*（arXiv:2102.09672）式 (5)，用余弦型 \(\bar\alpha\) 让高频段（接近 \(t=T\)）不再像 linear 那样过早把信号压到 0；工程实现（diffusers `cosine_beta_schedule`）为原始累乘 \(\tilde a_t=f(t)/f(0)\) 差分出 \(\beta\) 并 `clip(0, 0.999)`。

## 核心与易错点
1. **先差分再累乘**：cosine 给的闭式是 \(\tilde a_t\)（本身像 ᾱ），但按标准流程要转成 \(\beta_t=1-\tilde a_t/\tilde a_{t-1}\)、裁剪、**再** cumprod(1-β)。跳过裁剪时两者几乎一致，裁剪触发处才显差异——务必按题面口径。
2. **\(T+1\) 个采样点**：`linspace(0, T, T+1)` 覆盖 \(t=0..T\)，得到 \(T\) 个 β；`f(0)` 作归一化分母使 \(\tilde a_0=1\)。
3. **索引对齐**：输出 `alpha_bar[t-1]` 对应 \(\bar\alpha_t\)，长度恰为 \(T\)。
4. float32：`linspace` 指定 `dtype=torch.float32`。

## 对拍口径
参考解用 `linspace`+向量化差分/`cumprod`。判题 `_oracle` 用 Python `math` 逐 t 计算 β（cosine 逐点跑 \(f\)）并顺序累乘 `prev *= (1-b)`（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设单调不增、首项 <1、两调度值对拍等。

## 复杂度
\(O(T)\)。
