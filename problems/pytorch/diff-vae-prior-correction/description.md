# Diffusion：VAE latent 的 shift / scale 与先验校正

## 题目描述

潜空间扩散不在像素上工作，而在 VAE 的 latent 上工作，两者之间差一个**仿射换挡**。SDXL 用单一的 `scaling_factor`；SD3 / Flux 额外加 `shift_factor`（latent 的均值偏移不为 0）：

\[ \text{encode}:\quad z=\big((x-\text{shift})\cdot\text{scale}-\mu_c\big)/\sigma_c,\qquad \text{decode}:\quad x=\frac{z\cdot\sigma_c+\mu_c}{\text{scale}}+\text{shift} \]

其中 \(\mu_c,\sigma_c\) 是**每通道**的先验校正统计量（把 latent 白化成接近 \(N(0,1)\)，让训练用的噪声先验与推理时真正喂给模型的分布对齐）。顺序不能换：编码先减 `shift` 再乘 `scale`，解码先乘 \(\sigma_c\) 再除 `scale`、最后加 `shift`。

## 规则

给定 `latents`（至少二维，第 0 维 batch、第 1 维通道）、`scaling_factor`（必须为正）、`shift_factor`、可选的 `channel_mean` / `channel_std`（每通道一个值或标量，必须同时给或同时不给，\(\sigma_c>0\)）以及方向开关 `inverse`，返回 float32 张量。`inverse=False` 是解码方向（latent → 像素空间尺度），`inverse=True` 是编码方向。

## 示例

\[ z=\begin{bmatrix}0.5&-1.0\\2.0&0.0\end{bmatrix},\ \text{scale}=0.18215,\ \text{shift}=0.0609,\ \mu_c=[0.2,-0.3],\ \sigma_c=[1.5,0.5] \]

```
with stats -> [[ 5.276382, -4.331085], [17.628838, -1.586094]]
no stats   -> [[ 2.805890, -5.429081], [11.040861,  0.060900]]
```

## 约束

- `scaling_factor>0` 且有限；`shift_factor` 有限；`channel_std` 必须全为正。
- `channel_mean` / `channel_std` 成对出现；长度必须等于通道数（或为单元素标量），不允许二维。
- **恒等与往返**：`scale=1, shift=0` 且无统计量时两个方向都是恒等；任意参数下 `encode(decode(z))` 必须逐位还原 \(z\)。
- **每通道**：\(z=0\) 的通道映射到 \(\mu_c/\text{scale}+\text{shift}\)；通道内差分的放大倍数是 \(\sigma_c/\text{scale}\)。
