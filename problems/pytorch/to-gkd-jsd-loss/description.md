# GKD：广义 JSD 的在线蒸馏损失

## 题目描述

**GKD**（Agarwal et al. 2024, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, arXiv:2306.13649, ICLR 2024）解决的是序列级蒸馏里的**分布错配**：如果学生只在教师的离线输出上训练，推理时一旦自己走偏就再也回不来（exposure bias）。GKD 的做法是让学生在自己生成的序列上向教师学习，并用**广义 JSD** 度量两个 next-token 分布的距离：

$$D_\text{JSD}^{(\beta)}(p_t\,\|\,p_s)=\beta\,\mathrm{KL}(p_t\|m)+(1-\beta)\,\mathrm{KL}(p_s\|m),\qquad m=\beta\,p_t+(1-\beta)\,p_s .$$

$\beta$ 在「前向 KL」（覆盖教师所有模态）与「反向 KL」（只追教师的主模态）之间连续插值，比固定用某一种 KL 更可控。总损失再把蒸馏项与硬标签交叉熵按 $\lambda$ 混合：$L=(1-\lambda)D_\text{JSD}+\lambda L_\text{CE}$。

本题给定一批**逐位置的学生/教师 logits** 与硬标签，要求算出这条损失。

## 规则

给定 $N$ 个位置、词表大小 $V$：

1. **温度缩放后取对数概率**：$s=\log\text{softmax}(\text{student}/\tau)$，$t=\log\text{softmax}(\text{teacher}/\tau)$，$p_s=e^{s}$、$p_t=e^{t}$。
2. **混合分布按概率定义**：$m=\beta p_t+(1-\beta)p_s$；实现时用等价的对数式
   $\log m=\mathrm{logsumexp}\big([\,t+\log\beta,\ s+\log(1-\beta)\,]\big)$，避免 $m$ 过小造成下溢。
3. **广义 JSD**：$D=\beta\sum_v p_t(t-\log m)+(1-\beta)\sum_v p_s(s-\log m)$。
4. **交叉熵用未缩放的学生 logits**：$\text{CE}=-\log\text{softmax}(\text{student})[\text{label}]$。
5. 只在**有效标签**位置（`label != ignore_index`）上分别取均值，再合成
   $L=(1-\lambda)\,\overline{D}+\lambda\,\overline{\text{CE}}$，**不乘 $\tau^2$**。
6. $\lambda\in[0,1]$，$\beta$ **严格**在 $(0,1)$，$\tau>0$。返回 0 维 float64 标量。

## 示例

单位置、词表 2：

```
student = [2.0, 0.0]   teacher = [1.0, 0.0]   label = 0
λ = 0.5, β = 0.5, τ = 1.0

p_s = [0.880797, 0.119203]      p_t = [0.731059, 0.268941]
m   = 0.5·p_t + 0.5·p_s = [0.805928, 0.194072]
JSD = 0.5·KL(p_t‖m) + 0.5·KL(p_s‖m) = 0.018306
CE  = -log p_s[label=0]             = 0.126928
loss = 0.5 × 0.018306 + 0.5 × 0.126928 = 0.072617
```

把 $\lambda$ 置 0 只剩 $0.018306$、置 1 只剩 $0.126928$；把 $\tau$ 改成 0.5（分布更尖）损失变为 $0.074601$，说明温度确实只作用在 JSD 那一路。

## 约束

- 返回 0 维 float64；logits 可以是 float32，内部升到 float64。
- `ignore_index` 位置**同时**退出 JSD 与 CE 的均值（不是只传给 CE）；至少要有 1 个有效标签。
- $\beta$ 取 0 或 1 时上式整体退化为 0（混合权重把另一臂的 KL 系数也变 0），因此契约要求开区间；$\beta\to0$ 与 $JSD/\beta\to\mathrm{KL}(p_t\|p_s)$，$\beta\to1$ 与 $JSD/(1-\beta)\to\mathrm{KL}(p_s\|p_t)$。
- 非法输入抛 `ValueError`：非张量 / logits 非二维或形状不符 / 非浮点 / 含 NaN·Inf / $V<1$ / labels 非整型或形状不符 / 有效标签越界 / 无有效标签 / $\lambda$ 越界 / $\beta\notin(0,1)$ / $\tau\le0$。
- CPU、确定性、不调用随机数。
