# Diffusion：Rectified Flow 的 timestep shift 映射

## 题目描述

RF / Flow Matching 采样器的步长是均匀的，但高分辨率样本（token 数多）需要把更多步数花在**高噪段**——SD3 与 Flux 用一个只含一个参数的映射把整个时间栅格往噪声侧推：

\[ t' = \frac{s\,t}{1+(s-1)\,t} \;=\; \sigma\!\left(\operatorname{logit}(t)+\mu\right),\qquad s=e^{\mu} \]

也就是在 logit 域上平移 \(\mu\)。\(\mu\) 本身由图像序列长度线性插值得到（分辨率越高、序列越长，\(\mu\) 越大）：

\[ \mu = m\cdot\text{seq\_len}+b,\qquad m=\frac{\mu_{\max}-\mu_{\text{base}}}{\text{len}_{\max}-\text{len}_{\text{base}}},\qquad b=\mu_{\text{base}}-m\cdot\text{len}_{\text{base}} \]

## 规则

给定一维 `times` 与三者之一：`shift`（直接给 \(s\)）、`mu`（给 \(\mu\)，内部取 \(s=e^{\mu}\)）、`seq_len`（整数，按上式插值出 \(\mu\)）。其余参数 `base_shift=0.5`、`max_shift=1.15`、`base_seq_len=256`、`max_seq_len=4096` 可覆盖。返回 `(shifted, s)`：前者为与 `times` 同形状的 float32 张量，后者为 python float。

## 示例

\[ \text{seq\_len}=256\Rightarrow\mu=0.5,\ s=e^{0.5}=1.6487213,\ \text{times}=[0.25,0.5,0.75] \]

```
shifted = [0.354661, 0.622459, 0.831824]
s       = 1.6487212707
```

## 约束

- `times` 必须是一维 float 张量，全部落在 \([0,1]\) 内且不含 nan / inf。
- `shift` / `mu` / `seq_len` 必须**恰给一个**；`shift` 必须是正的有限实数，`seq_len` 必须是非负整数。
- **退化性质**：\(s=1\)（即 \(\mu=0\)）时映射是恒等；端点不动（\(t=0\to0\)、\(t=1\to1\)）；映射严格单调递增。
- **可加性**：先按 \(s_1\) 再按 \(s_2\) 平移，等价于一次 \(s_1s_2\)（因为 logit 域的平移可加）。
- \(s>1\) 把时间推向噪声侧（\(t'\ge t\)），\(s<1\) 推向干净侧；低于 `base_seq_len` 时按线性外推给出更小的 \(s\)（不截断）。
