# Adam Step Explained

## Update equations (compared with PyTorch)

Let $t$ be the step number (1-based), $g_t$ the gradient, $m$ the first moment (momentum), and $v$ the second moment (uncentered variance):

$$ m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2 $$
$$ \tilde m_t=\frac{m_t}{1-\beta_1^t},\quad \tilde v_t=\frac{v_t}{1-\beta_2^t},\quad p_t=p_{t-1}-\eta\frac{\tilde m_t}{\sqrt{\tilde v_t}+\epsilon} $$

PyTorch merges the two bias corrections into `step_size=lr/(1-β1^t)`, `denom=√v/√(1-β2^t)+eps`, and `p -= step_size*m/denom` — the reference solution for this problem uses exactly this form, matching the source code bit for bit.

## Why bias correction is needed

$m$ and $v$ are initialized to zero, so their expectation $\mathbb E[m_t]=(1-\beta_1)\sum\beta_1^{t-i}g_i$ is systematically too small early on (the geometric sum of $\beta_1$ weights is $<1$). The factor $1-\beta^t$ is precisely that weight sum, and dividing by it makes the estimate unbiased for small $t$. For large $t$, $\beta^t\to0$, so the correction approaches 1 and becomes negligible.

## Where eps goes

$\epsilon$ is added **after** the square root (in the denominator), not before it. Putting it in the wrong place significantly changes the update magnitude when gradients are near 0 and also disagrees with `torch.optim.Adam`.

## First-step intuition

$m_1=(1-\beta_1)g,\ v_1=(1-\beta_2)g^2$, so after correction $\tilde m/\sqrt{\tilde v}=\text{sign}(g)$, and the first step is exactly $p_1=p_0-\eta\,\text{sign}(g)$. The example [1,1]→[0.9,1.1] in the statement is exactly this property.

## The weight_decay pitfall (Adam ≠ SGD)

In SGD, weight_decay is equivalent to L2 regularization; but in Adam, adding $\lambda p$ to the gradient gets rescaled by the adaptive denominator, weakening the effect. **AdamW** instead uses "decoupled weight decay" $p\!-\!=\eta'\!p$ applied directly to the parameters. This problem uses amsgrad=False and no weight_decay, so that branch is not involved, but AdamW should be preferred when using Adam in production.

## Common mistakes

- Return **three** tensors `(new_params, new_exp_avg, new_exp_avg_sq)`; both states must be carried over correctly.
- `step` is 1-based; passing 0 makes $1-\beta^0=0$ and divides by zero.
- Do not modify the inputs in place (clone them); across a multi-step loop the states would contaminate each other.

## Related

sgd-momentum-step is the simplified version with only $m$; optimizer-training-loop shows how Adam is used in a real loop; problems such as top-p/gumbel do not involve optimizers.
