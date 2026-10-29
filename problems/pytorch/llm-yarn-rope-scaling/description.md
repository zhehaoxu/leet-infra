# YaRN / LongRoPE：波长 ramp 混合与注意力温度

## 题目描述

RoPE 的盲区在训练长度之外：低维（长波长）分量的角度会转过许多个 \(2\pi\)，模型从没见过这种组合。**YaRN**（Peng et al., 2309.00071，LongRoPE 采用同一套）不做一刀切的位置插值，而是按**波长**把维度切成三段：高频维（短波长、转得快）原样**外推**，低频维（长波长、转得慢）按 \(1/\text{factor}\) **插值**，中间用一个线性 ramp 过渡。两个边界由"转数"反解出来——`beta_fast` 给出外推侧的边界、`beta_slow` 给出插值侧的边界。

插值把所有角度压扁了，attention logits 随之变小、分布变尖，熵掉下来。YaRN 因此再补一个温度 \(\gamma = 0.1\ln s + 1\)，把它乘在 cos/sin 上。

本题要求一份实现同时覆盖三个档位：`yarn`（ramp 混合 + 温度）、`linear`（整张 inv_freq 除以 factor）、`dynamic_ntk`（按当前长度改写 base）。后两个是 YaRN 的退化对照，用来验证 ramp 的两端确实分别退化成它们。

## 规则

记 \(d=\text{head\_dim}\)（偶数，\(d\ge4\)）、\(h=d/2\)、\(i=0,\dots,h-1\)、\(L=\text{original\_max\_position\_embeddings}\)：

\[ \text{pos\_freqs}_i = \text{base}^{\,2i/d},\qquad \text{inv\_extrap}_i=\frac{1}{\text{pos\_freqs}_i},\qquad \text{inv\_interp}_i=\frac{\text{inv\_extrap}_i}{\text{factor}} \]

### yarn

**边界**：先按转数反解维度

\[ c(n)=\frac{d\cdot\ln\!\big(L/(n\cdot 2\pi)\big)}{2\ln\text{base}} \]

`truncate=True` 时 \(low=\lfloor c(\text{beta\_fast})\rfloor\)、\(high=\lceil c(\text{beta\_slow})\rceil\)；为 `False` 时保留浮点。再裁到 \(low=\max(low,0)\)、\(high=\min(high,d-1)\)；若 \(low=high\) 则令 \(high\mathrel{+}=10^{-3}\) 以免除零。

**ramp 与混合**：ramp 的自变量只铺满 \(h\) 个维度，而边界是在 \(d\) 的尺度上算出来的：

\[ \text{ramp}_i=\mathrm{clamp}\!\left(\frac{i-low}{high-low},0,1\right),\qquad \text{inv\_freq}_i=\text{inv\_interp}_i\cdot\text{ramp}_i+\text{inv\_extrap}_i\cdot(1-\text{ramp}_i) \]

于是 \(i<low\) 的维度完全外推、\(i>high\) 完全插值、中间线性过渡。因为两个端点都随 \(i\) 递减、权重又向更小的 `inv_interp` 移动，整张 `inv_freq` 单调不增。

**温度**：记 \(g(s,m)=1\)（\(s\le1\)）或 \(0.1\,m\ln s+1\)，则

\[ \gamma=\begin{cases}\text{attention\_factor}, & \text{显式给出}\\[2pt] g(\text{factor},\text{mscale})\,/\,g(\text{factor},\text{mscale\_all\_dim}), & \text{两个 mscale 都给}\\[2pt] g(\text{factor},1), & \text{其余}\end{cases} \]

### linear / dynamic_ntk

\[ \text{linear}:\ \text{inv\_freq}=\text{inv\_interp},\quad \gamma=1 \]

\[ \text{dynamic\_ntk}:\ \ell=\max(\text{seq\_len}\ \text{或}\ L,\ L),\quad \text{base}'=\text{base}\cdot\Big(\frac{\text{factor}\cdot\ell}{L}-(\text{factor}-1)\Big)^{d/(d-2)},\quad \text{inv\_freq}_i=\text{base}'^{-2i/d} \]

两者都没有温度，此时传入 `attention_factor` / `mscale` / `mscale_all_dim` 一律抛 `ValueError`。

### cos / sin

\[ \text{freqs}=\text{position\_ids}_{[\dots,\mathrm{None}]}\cdot\text{inv\_freq},\qquad \text{emb}=[\text{freqs}\ \Vert\ \text{freqs}] \]

\[ \cos=\cos(\text{emb})\cdot\gamma,\qquad \sin=\sin(\text{emb})\cdot\gamma \]

返回 `(inv_freq, gamma, cos, sin)`：`inv_freq` 形状 \((h,)\) float32，`gamma` 是 python float，`cos` / `sin` 形状 \((B,S,d)\) float32。

## 示例

取 \(d=8\)、`base=10000`、`factor=4`、\(L=2048\)，`position_ids=[[0,1,2,3]]`。此时 \(c(32)=1.008\)、\(c(1)=2.513\)，取整后 \(low=1\)、\(high=3\)，中间只有 \(i=2\) 落在 ramp 上（\(\text{ramp}=0.5\)）：

```
inv_freq    = [1.0, 0.1, 0.00625, 0.00025]          gamma = 1.138629
low = 1, high = 3    只有 i=2 落在 ramp 上：0.5*0.0025 + 0.5*0.01 = 0.00625
cos(pos=3)  = [-1.127235, 1.087774, 1.138429, 1.138629, ...]   后四维与前四维相同
sin(pos=3)  = [0.160683, 0.336488, 0.021348, 0.000854, ...]    gamma 同时乘在 cos/sin 上
```

\(i=0\) 完全外推（\(1.0\)）、\(i=1\) 也完全外推（\(0.1\)）、\(i=3\) 完全插值（\(0.001/4=0.00025\)）。温度 \(\gamma=0.1\ln4+1=1.138629\)：位置 0 的 cos 恰好等于 \(\gamma\) 而不是 1，这就是"把幅度当温度用"的直接后果——它不是旋转，而是把 logits 整体缩放 \(\gamma^2\)。

## 约束

- `head_dim` 必须是 \(\ge4\) 的偶数；`base` 必须 \(>1\)；`factor` 必须是有限正数；`original_max_position_embeddings` 是 \(\ge2\) 的整数（`bool` 不接受）。
- `beta_fast` / `beta_slow` 必须是有限正数且 **`beta_fast > beta_slow`**；`truncate` 必须是真正的 `bool`。
- `position_ids` 必须是二维整数张量的 \((B,S)\)、非空、非负。
- `rope_type` 只接受 `"yarn"` / `"linear"` / `"dynamic_ntk"`；`seq_len` 只在 `dynamic_ntk` 下使用，给出时必须是正整数（不足 \(L\) 自动抬到 \(L\)）。
- \(\gamma\) 同时乘在 cos **和** sin 上：旋转本身保范数，所以对任意向量施加本题的 cos/sin 后，其长度变为原来的 \(\gamma\) 倍。
- 内部用 float64 计算，返回 float32。
