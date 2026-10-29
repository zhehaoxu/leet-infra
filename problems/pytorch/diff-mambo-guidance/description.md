# 扩散采样：MAMBO-G 的幅值自适应引导

## 题目描述

大引导尺度下 CFG 最容易出问题的地方是**过饱和**：更新向量 \(\epsilon_c-\epsilon_u\) 的幅值一旦远大于无条件预测本身，样本就被推向色块化、纹理糊掉的区域。**Magnitude-Aware Mitigation for Boosted Guidance**（MAMBO-G）给出了一个几乎零成本的补丁——不改方向，只按「更新幅值相对于无条件预测幅值」的比例，把有效引导强度从 \(w\) 拉回 1：

\[ r=\frac{\lVert \epsilon_c-\epsilon_u\rVert_2}{\lVert \epsilon_u\rVert_2},\qquad w'=1+(w-1)\,e^{-\lambda r},\qquad \text{out}=\epsilon_u+w'\,(\epsilon_c-\epsilon_u) \]

比例 \(r\) 小时（更新相对温和）\(w'\approx w\)，行为与 CFG 一致；\(r\) 大时 \(w'\to 1\)，输出退回条件预测，相当于「这个样本已经被推得太狠，就别再推了」。\(\lambda\) 控制抑制的激进程度。

注意归约范围是**逐样本**的：对每个样本，除第 0 维（batch）以外的所有维度一起求 L2 范数，得到该样本的一个标量 \(r\)；不要按通道或按空间位置各自归一化。

## 规则

给定同形状的 `eps_cond` 与 `eps_uncond`（第 0 维是 batch）、基础引导强度 `guidance_scale = w ≥ 0` 与抑制强度 `alpha = λ ≥ 0`：

1. \(\delta=\epsilon_c-\epsilon_u\)；
2. 逐样本求 \(r=\lVert\delta\rVert_2/\lVert\epsilon_u\rVert_2\)；
3. \(w'=1+(w-1)e^{-\lambda r}\)；
4. \(\text{out}=\epsilon_u+w'\delta\)。

特殊情况：若某个样本的 \(\lVert\epsilon_u\rVert_2=0\)，把 \(r\) 视为 \(+\infty\)，于是 \(w'=1\)、输出恰为 \(\epsilon_c\)。这种情况不能出现 `NaN` 或 `inf`。

返回 float32 张量，形状与 `eps_cond` 相同。

## 示例

\[ \epsilon_c=\begin{bmatrix}1.5&-2.5&1.25\\-0.25&1.0&-1.5\end{bmatrix},\quad \epsilon_u=\begin{bmatrix}1.0&-2.0&1.0\\0.5&1.0&-1.0\end{bmatrix},\quad w=10 \]

```
alpha=0.0 -> [[ 6.000000, -7.000000,  3.500000], [-7.000000,  1.000000, -6.000000]]
alpha=0.5 -> [[ 5.361224, -6.361224,  3.180612], [-5.248210,  1.000000, -4.832140]]
alpha=2.0 -> [[ 3.939285, -4.939285,  2.469642], [-2.279302,  1.000000, -2.852868]]
```

两行的幅值比分别是 \(r\approx 0.3062\) 与 \(r\approx 0.6009\)，所以第二行被压得更狠；两行的 \(w'\) 都严格介于 \(1\) 与 \(10\) 之间，且 \(\lambda\) 越大越靠近 1。第二行第 2 个元素在两支预测里都等于 \(1\)，因此输出恒为 1。

## 约束

- 输入至少两维：第 0 维是 batch，其余维度全部参与范数归约；两支形状必须一致。
- `guidance_scale` 与 `alpha` 都是有限非负实数，`bool` 不算实数。
- **退化性质**：\(\lambda=0\) 时 \(e^{0}=1\)，输出恰为 \(\epsilon_u+w(\epsilon_c-\epsilon_u)\)（裸 CFG）；\(w=1\) 时 \(w'=1\)，输出与 `eps_cond` 逐位相同，与 \(\lambda\)、幅值比无关；\(w>1\) 时 \(w'\) 随 \(\lambda\) 单调递减，且 \(\lambda\to\infty\) 时收敛到 1。
- \(\lVert\epsilon_u\rVert_2=0\) 的样本输出 `eps_cond`，不得出现 `NaN`/`inf`（直接写 \(\lVert\delta\rVert/\lVert\epsilon_u\rVert\) 会得到 `inf`，再乘 0 就是 `NaN`）。
- 计算内部可用 float64，返回 float32。
