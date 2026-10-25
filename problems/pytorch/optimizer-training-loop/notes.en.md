# SGD Training Loop and Numerical Reproduction Explained

## Objective and hyperparameters

The scalar parameter $w$ starts at $2.0$ and minimizes $\text{loss}=w^2$; `SGD(lr=0.1, momentum=0.9)` runs for 20 steps, recording the loss **before** each update.

## The fixed five-step order (the lifeline of grading)

Each step strictly follows:

1. `opt.zero_grad()` — clear the gradients left over from the previous step (`.grad` accumulates; not clearing it means stacking).
2. `loss = (w**2).sum()` — build the forward graph.
3. `losses.append(loss.item())` — record **before backward/step**, so the $k$-th record is the loss before the $k$-th update, after the previous $k-1$ steps have been applied.
4. `loss.backward()` — backpropagate to fill `w.grad`.
5. `opt.step()` — update `w` according to the gradient.

A misordering (e.g. recording after step, or backward before zero_grad) shifts the entire loss curve, and element-wise grading will inevitably fail. `sum()` exists to turn the non-scalar $w^2$ into a scalar so it can be backpropagated directly.

## The math of momentum

The momentum method maintains a velocity buffer $v$:

$$ v_t=\mu\,v_{t-1}+\eta\,g_t,\qquad w_{t+1}=w_t-v_t $$

$\mu=0.9,\eta=0.1$. It adds inertia to the descent direction, oscillating across narrow valleys and accelerating convergence. In this problem $g_t=2w_t$ can be derived analytically, so the 20-step sequence is a **fully determined** float sequence — this is exactly the premise for bit-exact comparison (no random data, no dropout).

## Why it can be reproduced exactly

There is no data randomness and no difference in floating-point reduction order (a single scalar), and PyTorch's SGD+momentum update is deterministic. Therefore the loss at each step can be dumped verbatim into the problem statement as the expected value, without setting a seed.

## Related

autograd-basics provides the backward mechanism; nn-module-block provides the parameter container being optimized; dataset-dataloader is the "batched data" version of this loop.
