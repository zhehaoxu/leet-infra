# Adam 更新 详解

## 更新式（对照 PyTorch）

记步号 $t$（1-based）、梯度 $g_t$、一阶矩 $m$（动量）、二阶矩 $v$（未中心化方差）：

$$ m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2 $$
$$ \tilde m_t=\frac{m_t}{1-\beta_1^t},\quad \tilde v_t=\frac{v_t}{1-\beta_2^t},\quad p_t=p_{t-1}-\eta\frac{\tilde m_t}{\sqrt{\tilde v_t}+\epsilon} $$

PyTorch 把两处偏差修正合并成 `step_size=lr/(1-β1^t)`、`denom=√v/√(1-β2^t)+eps`、`p -= step_size*m/denom`——本题参考解用的就是这个写法，与源码逐位一致。

## 为什么有偏差修正

$m,v$ 以零初始化，其期望 $\mathbb E[m_t]=(1-\beta_1)\sum\beta_1^{t-i}g_i$ 早期系统性偏小（除以 $\beta_1$ 的几何权重和 $<1$）。$1-\beta^t$ 正是这个权重和，除掉它使 $t$ 较小时估计无偏。$t$ 大时 $\beta^t\to0$，修正趋于 1 可忽略。

## eps 的位置

$\epsilon$ 加在**开方之后**（分母），而不是开方之前。放错位置在梯度接近 0 时会显著改变更新幅度，也与 `torch.optim.Adam` 不符。

## 首步直觉

$m_1=(1-\beta_1)g,\ v_1=(1-\beta_2)g^2$，修正后 $\tilde m/\sqrt{\tilde v}=\text{sign}(g)$，故首步恰为 $p_1=p_0-\eta\,\text{sign}(g)$。题面示例 [1,1]→[0.9,1.1] 正是此性质。

## weight_decay 的坑（Adam ≠ SGD）

在 SGD 里 weight_decay 与 L2 正则等价；但在 Adam 里把 $\lambda p$ 加进梯度会被自适应分母重新缩放，效果打折。**AdamW** 改用「解耦权重衰减」$p\!-\!=\eta'\!p$ 直接作用于参数。本题 amsgrad=False、无 weight_decay，不涉及此分支，但生产用 Adam 时应优先 AdamW。

## 口径坑

- 返回**三个**张量 `(new_params, new_exp_avg, new_exp_avg_sq)`，两个状态都要 carry-over 正确。
- `step` 是 1-based；传 0 会让 $1-\beta^0=0$ 除零。
- 别原地改入参（clone）；多步循环里状态会互相污染。

## 关联

sgd-momentum-step 是只有 $m$ 的简化版；optimizer-training-loop 演示 Adam 在真实循环里的用法；top-p/gumbel 等题不涉及优化器。
