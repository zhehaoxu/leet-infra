# SGD with Momentum Explained

## Update Formulas (against PyTorch)

`torch.optim.SGD` (defaults `dampening=0, nesterov=False`) does each step:

$$ g_t \leftarrow \nabla f(p_t) + \lambda\, p_t \quad(\lambda=\text{weight\_decay}) $$
$$ b_t = \mu\, b_{t-1} + g_t \quad(\mu=\text{momentum}) $$
$$ p_{t+1} = p_t - \eta\, b_t $$

With momentum=0 there is no buffer and $p_{t+1}=p_t-\eta g_t$. The three steps cannot be reordered: **first** fold weight_decay into the gradient, **then** update the buffer, **finally** update the parameters with the buffer.

## Intuition

- **momentum**: view the gradient as acceleration and $b$ as velocity; the closer $\mu$ is to 1, the stronger the inertia. Along consistent directions the velocity accumulates (faster than plain SGD); in oscillating directions positive and negative cancel (damping oscillation). The effective learning rate is amplified to roughly $\eta/(1-\mu)$.
- **weight_decay vs L2**: here WD adds $\lambda p$ to the gradient, the original form of "decoupled weight decay" (for SGD, weight_decay ≡ L2, but not for Adam — see the pitfall in the adam notes).

## nesterov (not enabled in this problem)

The Nesterov version is $b_t=\mu b_{t-1}+g_t,\ p_{t+1}=p_t-\eta(g_t+\mu b_t)$, i.e. "look one step ahead, then correct". The judging setting is nesterov=False, so do not add that term.

## Specification Pitfalls

- **The buffer must start at 0**: PyTorch's first step gives `buf = grad` (since `0.9*0 + grad`). Mistakenly using None or not zeroing shifts everything.
- Return both `(new_params, new_buffers)`; the tests check whether the buffer carry-over is correct.
- Do not modify the input tensors in place (clone them), or repeated calls will pollute each other.

## Division of Labor with optimizer-training-loop

That problem is the "training-loop skeleton + per-step loss recording"; this one focuses on **the mathematics of a single update operator** — taking the optimizer apart from a black box into three algebraic steps.

## Related

adam-step is a more complex update with second moments + bias correction; the training problems such as weight-norm / batchnorm are all driven by step functions of this kind.
