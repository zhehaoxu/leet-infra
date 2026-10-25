# Adam Step

## Problem

Adam (Adaptive Moment Estimation) combines momentum (the first moment) with RMSProp (the second moment) and applies **bias correction** to both, making it the default optimizer for Transformers and most deep learning tasks. In this problem you hand-write a single Adam update: given the old parameters, gradients, the two running-average states, and the current step number, return all three updated values.

## Rules

- Implement `solve(params, grads, exp_avg, exp_avg_sq, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)`, returning `(new_params, new_exp_avg, new_exp_avg_sq)`.
- Follow **exactly the update equations of `torch.optim.Adam` (amsgrad=False, no weight_decay)** with $t=\text{step}$ (1-based):

  $$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
  $$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
  $$ \hat\epsilon_1 = 1-\beta_1^{t},\quad \hat\epsilon_2 = 1-\beta_2^{t} $$
  $$ \text{step\_size} = \text{lr}/\hat\epsilon_1,\quad \text{denom} = \sqrt{v_t}/\sqrt{\hat\epsilon_2} + \epsilon $$
  $$ p_t = p_{t-1} - \text{step\_size}\cdot m_t / \text{denom} $$

- The inputs must not be mutated (clone them); `exp_avg`/`exp_avg_sq` start as all zeros.

## Examples

```
Input: params=[1.0, 1.0], grads=[0.1, -0.2], exp_avg=[0,0], exp_avg_sq=[0,0], step=1, lr=0.1
new_params=[0.9, 1.1]
new_exp_avg=[0.01, -0.02]
new_exp_avg_sq=[1e-5, 4e-5]
(After bias correction the first step is equivalent to p - lr*sign(g), the signature behavior of Adam.)
```

## Constraints

- Support float32 / float64; the output dtype must match `params`
- Do not use `torch.optim.Adam`; the update must be written by hand and must match PyTorch step by step (see notes for the comparison)
