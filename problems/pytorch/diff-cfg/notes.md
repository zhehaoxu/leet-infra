# 详解：无分类器引导 (CFG)

## 出处
Ho & Salimans, *Classifier-Free Diffusion Guidance*（arXiv:2207.12598）。训练时以一定概率「丢掉」条件，使单网络能同时给 \(\epsilon_\theta(x_t,t,y)\)（有条件）与 \(\epsilon_\theta(x_t,t,\varnothing)\)（无条件）；采样时按权重线性外推。工程实现（diffusers `StableDiffusion`）用 `noise = noise_uncond + guidance_scale*(noise_cond - noise_uncond)`，本题采用该约定。

## 核心与易错点
1. **参数化差异**：diffusers 的 `guidance_scale=w` 与论文的 weight \(w'\) 差 1：\(w=w'+1\)。本题统一用 \(\epsilon_u+w(\epsilon_c-\epsilon_u)\)，\(w=1\) 即「无引导、纯条件」。
2. **方向是 cond−uncond**：外推沿「条件减无条件」方向，写反会把样本推向无条件。
3. **等价形式**：\(\epsilon_u+w(\epsilon_c-\epsilon_u)=(1-w)\epsilon_u+w\epsilon_c\)，两种写法数值一致，可作自检。
4. 结果可能溢出输入范围（\(w\) 大时正常），不做裁剪。

## 对拍口径
参考解 `eps_uncond + w*(eps_cond-eps_uncond)`。判题 `_oracle` 用等价式 `(1-w)*eps_uncond + w*eps_cond` 逐元素重算（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设 \(w=0\to\epsilon_u\)、\(w=1\to\epsilon_c\) 两个退化断言。

## 复杂度
逐元素 \(O(N)\)。
