# 扩散采样：AutoGuidance 的锚点与恒等阈值一起翻转

## 题目描述

AutoGuidance（*Guiding a Diffusion Model with a Bad Version of Itself*）不引入独立训练的空条件模型，而是把**同一个条件模型变坏**——采样时按概率丢弃若干 transformer block——把这份弱预测从条件预测旁边「推开」，从而放大强模型与弱模型的分布差。落到实现接口上，它复用的仍是 CFG 那套两支预测的签名：`pred_cond` 是完整模型的预测，`pred_uncond` 装的是弱模型预测。

diffusers 的 `AutoGuidance` 里有两条互相牵连的开关。第一条是 `use_original_formulation`：它同时改变**外推锚点**和**恒等（也就是自动关闭引导的）尺度**——默认实现把锚点放在弱预测上、以 \(s=1\) 为恒等；原论文表述把锚点放在条件预测上、以 \(s=0\) 为恒等。第二条是判定恒等所用的 `math.isclose`：与 \(1.0\) 相比走**相对**容差（默认 `rel_tol=1e-9`），与 \(0.0\) 相比只能走**绝对**容差（默认 `abs_tol=0`），所以「差一点点就算恒等」只在默认表述下成立。

本题要求实现这个单步算子，包括左闭右开的整型生效窗口，以及可选的标准差回缩（`guidance_rescale`）。

## 规则

给定两支同形状预测（第 0 维为 batch）：`pred_cond` \(\epsilon_c\)、`pred_uncond` \(\epsilon_w\)。尺度 \(s\) 即 `guidance_scale`，当前步 `step`，总步数 \(N\) 即 `num_inference_steps`，窗口 `start / stop`，表述开关 `use_original_formulation`，回缩系数 \(\phi\) 即 `guidance_rescale`。

生效判定，两项同时成立才算「开」：

\[ \text{within} = \big[\lfloor \text{start}\cdot N\rfloor \le \text{step} < \lfloor \text{stop}\cdot N\rfloor\big] \]

\[ \text{close} = \begin{cases}\text{isclose}(s, 0.0) & \text{use\_original\_formulation}\\[2pt]\text{isclose}(s, 1.0) & \text{否则}\end{cases}\qquad \text{on} = \text{within}\wedge\neg\text{close} \]

引导：

\[ \text{pred} = \begin{cases}\epsilon_c & \text{off}\\[3pt] \epsilon_c + s\,(\epsilon_c-\epsilon_w) & \text{on 且原论文表述}\\[3pt] \epsilon_w + s\,(\epsilon_c-\epsilon_w) & \text{on 且默认表述}\end{cases} \]

回缩（只要 \(\phi>0\) 就执行，**关闭分支同样执行**）：逐样本对除第 0 维以外的全部维度取无偏标准差（`torch.std` 默认，除 \(n-1\)），

\[ \text{pred} \leftarrow \phi\cdot \text{pred}\cdot\frac{\operatorname{std}(\epsilon_c)}{\operatorname{std}(\text{pred})} + (1-\phi)\cdot \text{pred} \]

关闭时 \(\epsilon_w\) 不参与计算，可以传 `None`（甚至传一张形状完全不符的张量）；生效却传 `None` 抛 `ValueError`。返回 float32 张量，形状与 `pred_cond` 相同。

三条最容易写错的地方：默认表述下 \(s=0\) **不是**恒等，它把结果整个搬到弱预测 \(\epsilon_w\) 上；原表述下 \(s=1\) 也不是恒等，它等于再外推一个 `shift`；\(s=1+10^{-10}\) 在默认表述下被判成恒等而关闭，\(s=10^{-12}\) 在原表述下却仍然开着。

## 示例

```
eps_c = [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, -1.0, 2.0]]
eps_w = [[0.2, 1.0, 2.5, 3.0], [0.5, 0.0, 0.5, -0.5]]
step = 5, N = 20, start = 0.1, stop = 0.8   # window = [2, 16), step 5 is inside
```

`phi = 0`（不做回缩）时：

```
s=7.5 native            -> [[ 6.200000,  8.500000,   6.250000, 10.500000], [-3.250000,  7.500000, -10.750000, 18.250000]]
s=7.5 original          -> [[ 7.000000,  9.500000,   6.750000, 11.500000], [-3.750000,  8.500000, -12.250000, 20.750000]]
s=1.0 native (off)      -> [[ 1.000000,  2.000000,   3.000000,  4.000000], [ 0.000000,  1.000000,  -1.000000,  2.000000]]
s=0.0 native            -> [[ 0.200000,  1.000000,   2.500000,  3.000000], [ 0.500000,  0.000000,   0.500000, -0.500000]]
s=0.0 original (off)    -> [[ 1.000000,  2.000000,   3.000000,  4.000000], [ 0.000000,  1.000000,  -1.000000,  2.000000]]
s=1.0 original          -> [[ 1.800000,  3.000000,   3.500000,  5.000000], [-0.500000,  2.000000,  -2.500000,  4.500000]]
```

两行的差恰好是一个 `shift`：`original - native = eps_c - eps_w`，与 \(s\) 无关。

`step=5`、\(s=7.5\)、默认表述、\(\phi=0.5\) 时，逐样本标准差为
\(\operatorname{std}(\epsilon_c)=[1.290994, 1.290994]\)、回缩前 \(\operatorname{std}(\text{pred})=[2.059682, 12.661252]\)：

```
s=7.5 native phi=0.5    -> [[ 5.043058,  6.913870,   5.083728,  8.540663], [-1.790692,  4.132366,  -5.923058, 10.055424]]
```

窗口外（`step=17`）或尺度落在恒等点时，输出与 `pred_cond` 逐位相同，且此时 `pred_uncond` 可以传 `None`。

## 约束

- `pred_cond` 必须是至少两维的 `torch.Tensor`（第 0 维为 batch），且除 batch 维以外元素个数 \(\ge 2\)；`pred_uncond`（生效时）形状必须与之一致。两者都必须全为有限值，否则抛 `ValueError`。
- `step`、`num_inference_steps` 为 Python `int`（`bool` 非法），\(N>0\) 且 \(0\le\text{step}<N\)。
- `start ∈ [0, 1)`（**严格小于 1**，对齐上游构造函数的校验），`stop ∈ [start, 1]`；`guidance_scale` 为有限实数；`guidance_rescale ∈ [0, 1]`；`use_original_formulation` 必须是 `bool`。违反任一约束抛 `ValueError`。
- `math.isclose` 用默认容差（`rel_tol=1e-9`、`abs_tol=0`），且比较对象是 \(0.0\) 还是 \(1.0\) 由 `use_original_formulation` 决定。
- 内部计算可用 float64，返回 float32。
- 禁止 `import diffusers`。
