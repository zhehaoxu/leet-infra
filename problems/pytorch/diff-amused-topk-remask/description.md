# AMUS 重掩码：把最不确定的 top-k 个位置再埋回去

## 题目描述

MaskGIT 一类的**离散扩散**（AMUS 用的就是这套）不把噪声加在连续值上，而是把 token 换成一个专门的 `mask_token_id`。采样时每一步都重新预测所有被掩位置，但只**保留**其中「最自信」的一部分，剩下的继续掩着，直到最后一步才全部接受。保留多少、留哪几个，就是这一步的全部算法。

AMUS 的选择判据不是原始概率，而是**加了 Gumbel 噪声的对数概率**：温度高的前几步允许随机性把一些其实还不错的位置也埋回去，避免早早锁死；温度退到 0 的末几步则近似「挑概率最低的几个」。同时每一步的掩码个数还要受两道钳位——不能比当前已掩的位置数还多（否则永远解不完），也不能少于 1（否则原地踏步）。

本题只做**一步**，并且完全不需要模型：softmax 的采样结果、Gumbel 噪声要的均匀随机数都由调用方显式传入，你只需要按下列规则算出这一步的输出。

## 规则

记 batch 为 \(B\)、序列长 \(S\)、词表 \(V\)、推理步数 \(N\)，`mask_token_id` 为 \(m\)。

1. **未知位置与候选 token**：\(\text{unknown}_i = (\text{sample}_i == m)\)，
   \[
   \text{committed} = \text{where}(\text{unknown},\ \text{drawn\_tokens},\ \text{sample})
   \]
   即被掩的位置用调用方采好的候选顶上，已解出的位置原样保留。
2. **步号换算**：调度器把 timestep 从 \(N-1\) 递减走到 \(0\)，而比例表按下标 \(0\dots N-1\) 递进，因此
   \[
   \text{step\_idx} = N - 1 - \text{timestep},\qquad \text{ratio} = \frac{\text{step\_idx} + 1}{N}
   \]
3. **掩码比例**（再乘 `starting_mask_ratio`）：
   \[
   \text{mask\_ratio} = \begin{cases}\cos(\text{ratio}\cdot \pi / 2) & \text{cosine}\\ 1 - \text{ratio} & \text{linear}\end{cases}
   \]
4. **两道钳位**（后两步逐 batch 行，`ratio` 对整批共享）：
   \[
   \text{mask\_len} = \big\lfloor S \cdot \text{mask\_ratio} \big\rfloor
   \;\xrightarrow{\ \min(\,\cdot,\ \sum \text{unknown} - 1\,)\ }\;
   \xrightarrow{\ \max(\,\cdot,\ 1\,)\ }
   \]
   下界 1 是**无条件**的：即使整行一个未知位置都没有，也要埋回 1 个。
5. **选中概率与哨兵**：\(p = \text{softmax}(\text{logits})\)，取 \(p_{\text{sel},i} = p_{i,\text{committed}_i}\)，然后
   \[
   p_{\text{sel}} \leftarrow \text{where}(\text{unknown},\ p_{\text{sel}},\ \texttt{finfo(float64).max})
   \]
   已解出的位置被顶到一个极大值，从而**不会**参与竞争。
6. **Gumbel 噪声与置信度**：设 \(u\) 为 `gumbel_uniforms`，温度序列 \(\text{temps} = \text{linspace}(\text{温度起点}, \text{温度终点}, N)\)，取 \(T = \text{temps}_{\text{step\_idx}}\)：
   \[
   g = -\ln\!\big(\operatorname{clamp}(-\ln(\operatorname{clamp}(u,\ 10^{-20})),\ 10^{-20})\big),\qquad
   c = \ln\!\big(\operatorname{clamp}(p_{\text{sel}},\ 10^{-20})\big) + T \cdot g
   \]
   两处 clamp 都要写：\(u = 0\) 给出 \(g \approx -3.829\)、\(u = 1\) 给出 \(g \approx 46.05\)，概率为 0 时按 \(10^{-20}\) 计。注意哨兵是**在取对数之前**替换概率，所以已知位置的置信度是 \(\ln(1.798\times 10^{308}) + T g \approx 709.78 + T g\)，而不是常数本身。
7. **严格小于第 mask\_len 小者**：把每行的 \(c\) 升序排序，取下标为 \(\text{mask\_len}\) 的那个作门槛，
   \[
   \text{masking}_i \iff c_i < c_{(\text{mask\_len})},\qquad
   \text{prev\_sample} = \text{where}(\text{masking},\ m,\ \text{committed})
   \]
   判据是**严格小于**，不是 \(\le\)：并列时谁都不会被掩，因此实际埋回的个数可以少于 \(\text{mask\_len}\)。
8. **末步收束**：\(\text{timestep} = 0\) 时不做任何重掩码——\(\text{prev\_sample} = \text{committed}\)，`masking` 全 False，`mask_len` 全 0。

返回 `(prev_sample, committed, masking, mask_len)`：前两个是 `int64`、形状 \((B, S)\)，`masking` 是 `bool`、形状 \((B, S)\)，`mask_len` 是 `int64`、形状 \((B, 1)\)。中间量一律用 `float64` 计算。

## 示例

\(B=1,\ S=6,\ V=3,\ m=2\)，其余入参：

```
logits (6 positions x 3 tokens):
  pos0: [ 2,  0, -1]   pos1: [ 0,  0,  0]   pos2: [ 1,  1,  0]
  pos3: [ 3,  0,  0]   pos4: [-1,  2,  0]   pos5: [ 0,  1,  2]
sample          = [[2, 0, 2, 2, 2, 2]]
drawn_tokens    = [[1, 0, 0, 2, 2, 0]]
gumbel_uniforms = [[0.2, 0.5, 0.9, 0.1, 0.75, 0.4]]
timestep = 2, num_inference_steps = 4, mask_token_id = 2
masking_schedule = "cosine", temperature = (2.0, 0.0), starting_mask_ratio = 1.0
```

`committed = [[1, 0, 0, 2, 2, 0]]`：位置 1 已解出（原值 0），其余五个位置用 `drawn_tokens` 顶上。
`step_idx = 4 - 1 - 2 = 1`，`ratio = 2/4 = 0.5`，`mask_ratio = cos(π/4) = 0.70711`，`floor(6 × 0.70711) = 4`；未知位置共 5 个，`min(4, 5-1) = 4`，`mask_len = [[4]]`。温度 `linspace(2, 0, 4)[1] = 4/3`，六个位置的置信度：

```
confidence = [-2.804359, 710.271397, 2.138495, -4.206966, -0.508647, -2.291044]
```

位置 1 是哨兵位。升序排列后取下标 4（即第 5 小）的 \(2.138495\) 作门槛，严格小于它的恰有 4 个位置——0、3、4、5——被埋回 \(m=2\)，而门槛本身所在的位置 2 被保留：

```
masking    = [[True, False, False, True, True, True]]
prev_sample = [[2, 0, 0, 2, 2, 2]]
```

同一组张量入参下只改参数：

- `timestep=3`（第一个去噪步）：`ratio=0.25`、`floor(6 × 0.92388)=5`，但被 `min(5, 5-1)=4` 压回 4，`masking` 与 `prev_sample` 同上；
- `timestep=1`：`ratio=0.75`、`mask_len=[[2]]`、`masking=[[True, False, False, True, False, False]]`、`prev_sample=[[2, 0, 0, 2, 2, 0]]`；
- `masking_schedule="linear"`、`timestep=2`：`mask_ratio=0.5`、`mask_len=[[3]]`、`masking=[[True, False, False, True, False, True]]`、`prev_sample=[[2, 0, 0, 2, 2, 2]]`；
- `masking_schedule="linear"`、`starting_mask_ratio=0.0`、`timestep=2`：`floor` 出 0 也被抬到 1，恰好埋回位置 3，`masking=[[False, False, False, True, False, False]]`、`prev_sample=[[1, 0, 0, 2, 2, 0]]`；
- `timestep=0`：`prev_sample == committed == [[1, 0, 0, 2, 2, 0]]`，`masking` 全 False，`mask_len=[[0]]`。

## 约束

- `logits` 必须是 `float32` / `float64` 的三维张量 \((B, S, V)\)，\(B \ge 1\)、\(S \ge 2\)、\(V \ge 1\)，且不含 nan / inf；
- `sample`、`drawn_tokens` 必须是 `int32` / `int64` 张量、形状 \((B, S)\)、取值落在 \([0, V)\)；
- `gumbel_uniforms` 必须是 `float32` / `float64` 张量、形状 \((B, S)\)、取值落在 \([0, 1]\)，不含 nan / inf；
- `timestep`、`num_inference_steps`、`mask_token_id` 必须是 Python 整数（布尔值不算）：\(N \ge 1\)、`timestep` 落在 \([0, N)\)、`mask_token_id` 落在 \([0, V)\)；
- `masking_schedule` 只能是字符串 `"cosine"` 或 `"linear"`；
- `temperature` 是实数，或恰含两个实数的元组/列表，绝对值都不超过 8（保证哨兵位严格高于任何未知位置）；单值 \(T\) 等价于 \((T, 0.01)\)；
- `starting_mask_ratio` 是落在 \([0, 1]\) 的有限实数；
- 违反以上任一条抛 `ValueError`；
- 性质：`mask_len` 恒在 \([1, S-1]\) 之间；只要 `sample` 里还有未知位置，已知位置的 `masking` 必为 False；`temperature=(0,0)` 时埋回的恰是「选中概率最低的 `mask_len` 个未知位置」；同行内置信度并列时（例如整行已知且 `gumbel_uniforms` 相同）`masking` 可以全 False，埋回个数少于 `mask_len`。
