# Adam 更新单步（Adam Step）

## 题目描述

Adam（Adaptive Moment Estimation）把动量（一阶矩）与 RMSProp（二阶矩）结合，并对两者做**偏差修正**，是 Transformer 与大多数深度任务的首选优化器。本题手写一次 Adam 更新——给定旧参数、梯度、两个滑动平均状态与当前步号，返回更新后的三者。

## 规则

- 实现 `solve(params, grads, exp_avg, exp_avg_sq, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)`，返回 `(new_params, new_exp_avg, new_exp_avg_sq)`。
- **严格按 `torch.optim.Adam`（amsgrad=False, 无 weight_decay）的更新式**（$t=\text{step}$，1-based）：

  $$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
  $$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
  $$ \hat\epsilon_1 = 1-\beta_1^{t},\quad \hat\epsilon_2 = 1-\beta_2^{t} $$
  $$ \text{step\_size} = \text{lr}/\hat\epsilon_1,\quad \text{denom} = \sqrt{v_t}/\sqrt{\hat\epsilon_2} + \epsilon $$
  $$ p_t = p_{t-1} - \text{step\_size}\cdot m_t / \text{denom} $$

- 入参不可变（clone）；`exp_avg`/`exp_avg_sq` 初值为全零。

## 示例

```
输入：params=[1.0, 1.0]，grads=[0.1, -0.2]，exp_avg=[0,0]，exp_avg_sq=[0,0]，step=1，lr=0.1
new_params=[0.9, 1.1]
new_exp_avg=[0.01, -0.02]
new_exp_avg_sq=[1e-5, 4e-5]
（首步偏差修正后等价于 p - lr*sign(g)，是 Adam 的标志性行为）
```

## 约束

- 支持 float32 / float64；输出 dtype 与 `params` 一致
- 不使用 `torch.optim.Adam`；需手写更新式，须与 PyTorch 逐步一致（对拍见 notes）
