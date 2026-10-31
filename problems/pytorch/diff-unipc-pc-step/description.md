# 扩散采样：UniPC 的预测-校正单步，两侧历史不同源

## 题目描述

`UniPCMultistepScheduler.step()` 在一次调用里做了两件事：先拿**当前**的网络输出回头**校正**上一步的预测，再用旋转过的历史向**前预测**一步。本题要求的就是这一对更新，输入已经是调度器内部口径的张量（`predict_x0=True` 时是 \(x_0\) 预测，`False` 时是 ε 预测），不涉及真实网络。

难点全在两处「不对称」上。其一，**两侧看到的最近历史不是同一个东西**：上游在校正之后才把缓冲左移，所以校正侧的 \(m_0\) 是上一步的输出、更老的项从 `m_hist[-2]` 起取，而预测侧的 \(m_0\) 是**当前**输出 `m_t`、更老的项从 `m_hist[-1]` 起取；两侧对应的 \(\lambda\) 下标也各差一格。其二，**被钉成 0.5 的阶数不同**：预测侧在 `order_p == 2` 时跳过线性求解、校正侧在 `order_c == 1` 时跳过。又因为 \(b_i\propto 1/B_h\)，\(B_h\cdot\text{rhos}\) 在走通式时与 `solver_type` 无关——于是 `bh1`/`bh2` 只移动那两支被钉住的项，其余配置逐位相同。

## 规则

所有中间量在 float64 下计算，两个返回值转成 float32。记 `s = sigmas`。

**调度量**：`use_flow_sigmas=False`（VP）时 \(\alpha_i=1/\sqrt{s_i^2+1},\ \tilde\sigma_i=s_i\alpha_i\)；为真（flow）时 \(\alpha_i=1-s_i,\ \tilde\sigma_i=s_i\)。再取 \(\lambda_i=\log\alpha_i-\log\tilde\sigma_i\)。

**一侧的通用件**（\(r\) 为该侧阶数，基准下标 \(s_0\)、目标下标 \(t\)，\(h=\lambda_t-\lambda_{s_0}\)）：

\[ hh=\begin{cases}-h & \text{predict\\_x0}\\[2pt] h & \text{otherwise}\end{cases}\qquad \varphi_1=\operatorname{expm1}(hh)\qquad B_h=\begin{cases}hh & \text{bh1}\\ \operatorname{expm1}(hh) & \text{bh2}\end{cases} \]

对 \(i=1\dots r-1\) 取该侧的历史 \(m_i\) 与下标 \(s_i\)，令 \(rk_i=(\lambda_{s_i}-\lambda_{s_0})/h\)、\(D_1^{(i)}=(m_i-m_0)/rk_i\)；把 \(rk_r=1\) 追加到末尾，\(R\) 的第 \(i\) 行为 \((rk_1^{\,i-1},\dots,rk_r^{\,i-1})\)。标量序列按

\[ v_1=\frac{\varphi_1}{hh}-1,\quad f_1=1,\qquad b_i=\frac{v_i f_i}{B_h},\quad f_{i+1}=(i+1)f_i,\quad v_{i+1}=\frac{v_i}{hh}-\frac{1}{f_{i+1}} \]

生成——注意上游是**先更新 \(f\) 再算下一条 \(v\)**，写成 \(v_{i+1}=v_i/hh-1/f_i\) 会错。

**系数 \(\text{rhos}\)**：校正侧 `order_c == 1` 时取 `[0.5]`，否则为 \(\operatorname{solve}(R,b)\)；预测侧 `order_p == 2` 时取 `[0.5]`、`order_p == 1` 时为空，否则为 \(\operatorname{solve}(R_{<r,\,<r},\,b_{<r})\)，即**同时划掉最后一行和最后一列**。

**校正侧**（`x_last` 为 None 时整侧跳过）：\(s_0=\)`step_index` \( -1\)、\(t=\)`step_index`；\(m_0=\)`m_hist[-1]`，\(m_i=\)`m_hist[-(i+1)]`，\(s_i=\)`step_index` \(-(i+1)\)。以 \(m_t=\)`m_t` 补上最后一项，

\[ c=\sum_{i<r}\text{rhos}_i D_1^{(i)}+\text{rhos}_r\,(m_t-m_0) \]

**预测侧**：\(s_0=\)`step_index`、\(t=\)`step_index` \(+1\)；\(m_0=m_t\)，\(m_i=\)`m_hist[-i]`，\(s_i=\)`step_index` \( - i\)，\(p=\sum_{i}\text{rhos}_i D_1^{(i)}\)（项数与 \(D_1\) 相同）。

**更新式**（两侧同形，\(x_{s_0}\) 在校正侧是 `x_last`、在预测侧是校正结果或 `x_this`）：

\[ x_t=\begin{cases}\dfrac{\tilde\sigma_t}{\tilde\sigma_{s_0}}\,x_{s_0}-\alpha_t\varphi_1 m_0-\alpha_t B_h(\text{差分组合}) & \text{predict\\_x0}\\[6pt] \dfrac{\alpha_t}{\alpha_{s_0}}\,x_{s_0}-\tilde\sigma_t\varphi_1 m_0-\tilde\sigma_t B_h(\text{差分组合}) & \text{otherwise}\end{cases} \]

返回二元组 `(corrected, prev)`；`x_last` 为 None 时 `corrected` 就是 `x_this` 原样透传。

## 示例

`sigmas = [8, 4, 2, 1, 0.5, 0.25, 0.125]`（VP 口径）的调度表关于 \(\lambda=0\) 完全对称，`step_index=3` 时两侧步长都是 \(h=0.693147\)、\(hh=-0.693147\)：

```
i     sigma      alpha   sigma~      lambda
0   8.000000  0.124035  0.992278   -2.079442
1   4.000000  0.242536  0.970143   -1.386294
2   2.000000  0.447214  0.894427   -0.693147
3   1.000000  0.707107  0.707107    0.000000
4   0.500000  0.894427  0.447214    0.693147
5   0.250000  0.970143  0.242536    1.386294
6   0.125000  0.992278  0.124035    2.079442
```

`order_c=2, order_p=2, solver_type="bh2"` 时两侧中间量（\(rk_r=1\) 已列在末尾）：

```
corrector  rks=[-1.000000, 1.000000]  phi1=-0.500000  B_h=-0.500000
           R=[[1.000000, 1.000000], [-1.000000, 1.000000]]
           b=[0.557305, 0.391958]     rhos=[0.082674, 0.474631]
predictor  rks=[-1.000000, 1.000000]  phi1=-0.500000  B_h=-0.500000
           b=[0.557305, 0.391958]     rhos=[0.5]        (pinned, no solve)
```

给定

```
m_hist[0] = [[ 0.4, -0.2], [ 1.1, 0.3]]
m_hist[1] = [[-0.5,  0.7], [ 0.2, 0.9]]
m_hist[2] = [[ 0.6,  0.1], [-0.3, 0.5]]
m_t       = [[ 0.2, -0.4], [ 0.8, 0.15]]
x_last    = [[ 1.3, -0.9], [ 0.4, 1.7]]
x_this    = [[ 1.1, -0.7], [ 0.6, 1.5]]
```

`step_index=3`、`predict_x0=True`、其余取默认时：

```
(2,2) bh2     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.762047, -0.782485], [0.844157, 0.906083]]
(2,2) bh1     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.727496, -0.825674], [0.939173, 0.875850]]
(2,3) bh2     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.433403, -0.774073], [1.211966, 0.907726]]
(2,3) bh1     corrected=[[1.204902, -0.777599], [0.380135, 1.450320]]  prev=[[0.433403, -0.774073], [1.211966, 0.907726]]
(1,1) bh2     corrected=[[1.169162, -0.764545], [0.404616, 1.458873]]  prev=[[0.828885, -0.662426], [0.613673, 0.989754]]
(1,1) bh1     corrected=[[1.141846, -0.798689], [0.479733, 1.434972]]  prev=[[0.811610, -0.684021], [0.661180, 0.974638]]
```

`(2,2)` 两行的 `corrected` 相同而 `prev` 差 0.095016，`(1,1)` 两行则 `corrected` 就差了 0.075117：前者是预测侧钉 0.5、后者是校正侧钉 0.5。`(2,3)` 两行完全一致，因为两侧都走通式。`predict_x0=False` 的 `(2,3)` 给出

```
corrected=[[1.668741, -1.331373], [0.593786, 2.446867]]  prev=[[2.342067, -1.422294], [-0.082072, 3.089230]]
```

把 `x_last` 置为 None 时 `corrected` 原样返回 `x_this`，`prev` 由 `x_this` 直接外推。

## 约束

- `m_t / m_hist[*] / x_this`（以及非 None 的 `x_last`）必须同形状、至少二维（第 0 维为 batch）、全为有限值；`m_hist` 是非空序列。
- `sigmas` 是一维、有限、**严格为正**的张量（终端 \(\sigma=0\) 的那一步不在本题范围内：此时 `bh1` 会给出 NaN 或 ±inf），且 `len(sigmas) >= step_index + 2`。
- `step_index / order_c / order_p` 是 Python int（`bool` 非法），`order_c, order_p >= 1`；启用校正时需 `step_index >= order_c` 且 `len(m_hist) >= order_c`，任何情况下都需 `step_index >= order_p - 1` 且 `len(m_hist) >= order_p - 1`。
- `use_flow_sigmas / predict_x0` 是 bool；flow 口径下 \(\sigma\ge1\) 会让 \(\lambda\) 失去有限性，抛 `ValueError`。
- `solver_type` 只认 `bh1` / `bh2`；上游把 `midpoint` / `heun` / `logrho` 静默映射成 `bh2`，本题保留该映射，其余名字抛 `ValueError`。
- 禁止 `import diffusers`。
