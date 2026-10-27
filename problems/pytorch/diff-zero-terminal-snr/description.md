# Diffusion：zero-terminal-SNR 重标定与 log-SNR 参数化

## 题目描述

原始扩散计划末端 \(\bar\alpha_T=\prod(1-\beta_i)\) 通常仍是 \(10^{-4}\) 量级，于是终点的 \(\mathrm{SNR}=\bar\alpha/(1-\bar\alpha)\) 远大于 0：模型从来见不到「纯噪声」，推理第一步就落在一个训练没覆盖的区域。zero-terminal-SNR（Lin et al.）的做法是把整条计划重标定成端点 \(\bar\alpha'_T=0\)：

\[ \bar\alpha'_t=\frac{\bar\alpha_t-\bar\alpha_T}{1-\bar\alpha_T} \;\Longrightarrow\; 1-\bar\alpha'_t=\frac{1-\bar\alpha_t}{1-\bar\alpha_T} \]

即把 \((1-\bar\alpha)\) 整体放大 \(1/(1-\bar\alpha_T)\) 倍，于是 \(\bar\alpha'_0=1\)（当 \(\beta_0=0\)）而 \(\bar\alpha'_T=0\) 精确成立。随后一切都在 log-SNR 域记账：

\[ \lambda_t=\log\frac{\bar\alpha'_t}{1-\bar\alpha'_t}\in(-\infty,+\infty],\qquad \theta_t=\operatorname{atan2}\!\left(\sqrt{1-\bar\alpha'_t},\sqrt{\bar\alpha'_t}\right) \]

\(\theta\) 是 \(\epsilon\to v\) 的旋转角：\(v=\cos\theta\,\epsilon-\sin\theta\,x_0\)。末端 \(\theta=\pi/2\) 时 \(v=-x_0\)，不再需要除以 \(\alpha=0\)——这正是零终端 SNR 必须搭配 v-prediction 的原因。

## 规则

给定一维 `betas`（元素在 \([0,1)\)，索引 0 最干净、末位最噪声），返回 `(alpha_bars, log_snrs, angles)` 三个同长度 float32 张量：重标定后的 \(\bar\alpha'\)、\(\lambda\)、\(\theta\)。两端分别取 \(\lambda=-\infty\)（\(\bar\alpha'=0\)）与 \(+\infty\)（\(\bar\alpha'=1\)），角度取 \(\pi/2\) 与 \(0\)。

## 示例

\[ \text{betas}=[0.1,0.2,0.4]\Rightarrow\bar\alpha=[0.9,0.72,0.432],\ \bar\alpha_T=0.432 \]

```
alpha_bars = [0.823944, 0.507042, 0.0]
log_snrs   = [1.543298, 0.028171, -inf]
angles     = [0.432994, 0.778356, 1.570796]
```

## 约束

- `betas` 每个元素必须在 \([0,1)\) 内且不含 nan/inf；空张量、非一维张量直接抛 `ValueError`。
- 全零 `betas` 会让 \(\bar\alpha_T=1\)，重标定变成 \(0/0\)，必须抛 `ValueError`。
- **退化性质**：\(\bar\alpha'\) 单调不增、\(\lambda\) 单调不增、\(\theta\) 单调不减；末端 \(\bar\alpha'\) 精确为 0（不是 \(10^{-7}\)），前一项必须仍为正。
- \(\theta\) 与 \(\lambda\) 满足闭式 \(\theta=\arctan(e^{-\lambda/2})\)；\(\cos\theta=\alpha\)、\(\sin\theta=\sigma\)。
