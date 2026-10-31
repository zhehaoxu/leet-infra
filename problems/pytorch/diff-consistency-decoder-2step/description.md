# 扩散采样：Consistency Decoder 的调度表与两步去噪

## 题目描述

diffusers 的 `ConsistencyDecoderScheduler` 服务的是「一致性解码器」：网络每一步输出的就是干净的 \(x_0\) 预测，采样过程只有**两步**——第一步去噪后必须**重新加噪**到第二个时间点，第二步去噪后直接交出结果。它不接收外部的 \(\bar\alpha\) 表，而是在构造期用 `betas_for_alpha_bar` 现算一张长度 \(T\) 的调度表，再由这张表导出 `c_skip / c_out / c_in` 三组系数。

本题把这两段合在一起考：先按指定变换重建整张调度表，再跑完这两步。网络的两次输出以显式入参给出，正态噪声同样显式给出，全程不涉及真实网络与随机数生成。

## 规则

**调度表**（所有中间量在 float64 下计算）：

按 `alpha_transform` 取 \(\operatorname{ab}(t)\)：

- `cosine`：\(\operatorname{ab}(t)=\cos\!\big(\frac{t+0.008}{1.008}\cdot\frac{\pi}{2}\big)^2\)
- `laplace`：\(\lambda=-\frac12\cdot\operatorname{copysign}(1,\,0.5-t)\cdot\ln(1-2|0.5-t|+10^{-6})\)，\(S=e^{\lambda}\)，\(\operatorname{ab}(t)=\sqrt{S/(1+S)}\)
- `exp`：\(\operatorname{ab}(t)=e^{-12t}\)

对 \(i=0\dots T-1\) 取

\[ \beta_i=\min\!\Big(1-\frac{\operatorname{ab}((i+1)/T)}{\operatorname{ab}(i/T)},\ \text{max\_beta}\Big),\qquad \bar\alpha_i=\prod_{j\le i}(1-\beta_j),\qquad \sigma_i=\sqrt{\frac{1}{\bar\alpha_i}-1} \]

\[ c_{\text{skip}}^{(i)}=\bar\alpha_i^{-1/2}\cdot\frac{\sigma_d^2}{\sigma_i^2+\sigma_d^2},\qquad c_{\text{out}}^{(i)}=\frac{\sigma_i\,\sigma_d}{\sqrt{\sigma_i^2+\sigma_d^2}},\qquad c_{\text{in}}^{(i)}=\frac{\bar\alpha_i^{-1/2}}{\sqrt{\sigma_i^2+\sigma_d^2}} \]

**两步更新**，记两个时间点为 \(t_0=\)`timesteps[0]`、\(t_1=\)`timesteps[1]`：

\[ \text{scaled\_first}=x_{\text{init}}\cdot c_{\text{in}}^{(t_0)} \]
\[ x_0^{(1)}=c_{\text{out}}^{(t_0)}\cdot F_1+c_{\text{skip}}^{(t_0)}\cdot x_{\text{init}} \]
\[ x_{\text{second}}=\sqrt{\bar\alpha_{t_1}}\;x_0^{(1)}+\sqrt{1-\bar\alpha_{t_1}}\;\cdot\text{noise} \]
\[ \text{scaled\_second}=x_{\text{second}}\cdot c_{\text{in}}^{(t_1)},\qquad x_{\text{final}}=c_{\text{out}}^{(t_1)}\cdot F_2+c_{\text{skip}}^{(t_1)}\cdot x_{\text{second}} \]

三个易错点：其一，重加噪用的是**下一个**时间点 \(t_1\) 的 \(\sqrt{\bar\alpha}/\sqrt{1-\bar\alpha}\)，不是当前步 \(t_0\) 的；其二，末步**不再加噪**，\(x_{\text{final}}\) 直接就是 \(x_0\) 预测，所以 `noise` 只通过 \(x_{\text{second}}\) 影响结果一次；其三，\(c_{\text{skip}},c_{\text{in}}\) 里的 \(\bar\alpha^{-1/2}\) 是**倒数**平方根（即 \(x_0\) 尺度上的反归一化），写成 \(\sqrt{\bar\alpha}\) 会把系数整体压低。

返回四元组 `(scaled_first, x_second, scaled_second, x_final)`，都是与 `x_init` 同形状的 float32 张量。

## 示例

`T=1024`、`alpha_transform="cosine"`、`sigma_data=0.5`、`timesteps=(1008, 512)`、`max_beta=0.999` 时，两个时间点上的调度量为：

```
t=1008  abar=5.210667e-04  sqrt_abar=0.022827  sqrt_1_minus_abar=0.999739
        sigma=43.7966  c_skip=0.005709  c_out=0.499967  c_in=1.000195
t=512   abar=4.923217e-01  sqrt_abar=0.701656  sqrt_1_minus_abar=0.712515
        sigma=1.0155  c_skip=0.278100  c_out=0.448573  c_in=1.259124
```

给定

```
x_init = [[0.8, -0.3], [1.2, 0.1], [0.0, 1.5]]
f_first = [[-0.5, 1.0], [0.25, -0.75], [1.0, 0.5]]
f_second = [[0.3, -0.2], [-0.6, 0.9], [0.15, 0.45]]
noise = [[1.0, -1.0], [0.5, 0.0], [-0.25, 0.75]]
```

`num_train_timesteps=1024`（其余取默认）时：

```
scaled_first  = [[0.800156, -0.300059], [1.200235, 0.100020], [0.000000, 1.500293]]
x_second      = [[0.540317, -0.362912], [0.448766, -0.262703], [0.172676, 0.715798]]
scaled_second = [[0.680326, -0.456951], [0.565052, -0.330776], [0.217421, 0.901278]]
x_final       = [[0.284834, -0.190640], [-0.144342, 0.330657], [0.115307, 0.400921]]
```

`cosine` 的 `beta_1023` 原始值是 \(1.0\)，被 `max_beta=0.999` 截断；把 `max_beta` 压到 \(0.05\) 会同时改变 \(c_{\text{in}}\) 与重加噪系数，四段输出全部移动。

## 约束

- `x_init / f_first / f_second / noise` 必须同形状（第 0 维为 batch）、全为有限值；否则抛 `ValueError`。
- `num_train_timesteps` 是 Python int（`bool` 非法）且 `>= 2`；`timesteps` 是**恰好两个** Python int，落在 \([0, T)\) 且严格递减；`sigma_data > 0`；`max_beta ∈ (0, 1]`；`alpha_transform` 只能是三个已知名字之一。
- 若调度表退化到某个 \(\bar\alpha_i \le 0\)（例如 `cosine` 配 `max_beta=1.0` 时尾步 \(\beta\) 触顶），\(\sigma_i\) 会发散，抛 `ValueError`。
- 中间量用 float64，四段返回都是 float32。禁止 `import diffusers`。
