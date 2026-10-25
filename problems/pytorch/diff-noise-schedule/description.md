# Diffusion：线性 / 余弦噪声调度 ᾱ 序列

## 题目描述

前向加噪 \(q(x_t|x_0)=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\) 的全部系数都由一个标量序列 \(\bar\alpha_t\)（信号在 \(t\) 步后的保留比例）决定。它来自逐步「剩余方差」\(\beta_t\) 的乘积：\(\alpha_t=1-\beta_t\)，\(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\)。本题按两种主流 \(\beta\) 调度生成 \(\bar\alpha\) 序列。

- **linear**（Ho et al. DDPM）：\(\beta_t\) 从 \(1\times10^{-4}\) 线性升到 \(0.02\)，共 \(T\) 步，即 `linspace(1e-4, 0.02, T)`。
- **cosine**（Nichol & Dhariwal, arXiv:2102.09672）：令偏移 \(s=0.008\)、\(f(u)=\cos\!\bigl(\tfrac{u/T+s}{1+s}\cdot\tfrac{\pi}{2}\bigr)^2\)，对 \(t=0,1,\dots,T\) 取原始累乘 \(\tilde a_t=f(t)/f(0)\)（\(\tilde a_0=1\)），再 \(\beta_t=1-\tilde a_t/\tilde a_{t-1}\) 并**裁剪到 \([0,0.999]\)**。

两种调度都先得到长度 \(T\) 的 \(\beta\)，再 \(\bar\alpha_t=\prod_{s\le t}(1-\beta_s)\)。

## 规则

输入正整数 `T` 与字符串 `kind`（`"linear"` 或 `"cosine"`）。返回长度 \(T\) 的 float32 张量 `alpha_bar`，其中 `alpha_bar[t-1]` = \(\bar\alpha_t\)。禁止现成扩散库。

## 示例

`T=4, kind="linear"`：
```
alpha_bar = [0.9998999834060669, 0.9931672811508179, 0.9798920154571533, 0.9602941870689392]
```
`T=4, kind="cosine"`：
```
alpha_bar = [0.8470122218132019, 0.49384361505508423, 0.1442720741033554, 0.00014427020505536348]
```

## 约束

- `T>=2`，float32；`kind` 只取 `"linear"` / `"cosine"`。
- `alpha_bar` 单调不增且首项 `<1`（\(\bar\alpha_1=1-\beta_1\)）。
- cosine 需先按 \(t=0..T\) 生成 \(T+1\) 个 \(\tilde a\)、逐段差分算 \(T\) 个 \(\beta\) 并裁剪，再乘回 \(\bar\alpha\)（裁剪后的 \(\beta\) 才用于累乘）。
