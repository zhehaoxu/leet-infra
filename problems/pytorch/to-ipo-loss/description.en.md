# IPO: Identity Preference Optimization Squared Loss

## Problem

**IPO** (Azar et al. 2024, *A General Theoretical Paradigm to Understand Learning from Human Preferences*, arXiv:2310.12036) points out a theoretical defect of DPO's logistic preference loss: it keeps supplying gradient on pairs where chosen already clearly beats rejected, pushing the implicit reward difference ever larger and finally overfitting the preference data. IPO's fix is blunt — pull the implicit reward difference $\Delta$ toward a **fixed target margin** $1/(2\beta)$ and replace sigmoid with a squared loss:

$$\mathcal{L}_\text{IPO}=\mathbb{E}\Big[\big(\Delta-\tfrac{1}{2\beta}\big)^2\Big],\qquad \Delta=(\log\pi_\theta(y_w)-\log\pi_\text{ref}(y_w))-(\log\pi_\theta(y_l)-\log\pi_\text{ref}(y_l)).$$

Once $\Delta$ reaches the target the gradient is exactly zero, so the model neither keeps widening the gap nor oscillates around the target.

This problem gives four **sequence-level** logprobs (chosen/rejected × policy/reference); you only assemble $\Delta$ and the square.

## Rules

Given a batch of length $N\ge 1$ with four sequence-level logprobs, each shaped $(N,)$ and sharing one dtype:

1. Per sample compute the implicit reward difference
   $$\Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l).$$
   The inputs are already sums over the whole answer, so **no length normalization** is applied.
2. Loss $=\frac{1}{N}\sum(\Delta-\frac{1}{2\beta})^2$, with $\beta>0$.
3. **Do not scale by $\beta$**: unlike DPO's $-\log\sigma(\beta\Delta)$, $\beta$ enters only through the target margin $1/(2\beta)$; a global rescale changes the loss magnitude but not the optimum.
4. Return a 0-dimensional float64 scalar.

## Examples

```
logp_w = [-1.0, -2.0]   ref_w = [-1.2, -1.8]
logp_l = [-3.0, -2.6]   ref_l = [-2.8, -2.4]
β = 0.5  ->  target margin 1/(2β) = 1.0

sample 0: Δ = (-1.0 + 1.2) - (-3.0 + 2.8) = 0.2 - (-0.2) = 0.4
sample 1: Δ = (-2.0 + 1.8) - (-2.6 + 2.4) = -0.2 - (-0.2) = 0.0
loss = mean([(0.4-1)², (0.0-1)²]) = mean([0.36, 1.0]) = 0.68
```

Scaling the whole loss by $\beta$ would give 0.34 at $\beta=0.5$; writing the target as $1/\beta$, or omitting the square, also fails to reproduce the number.

## Constraints

- Return a 0-dimensional float64 scalar; float32 inputs are promoted to float64 internally.
- The loss is exactly 0 when $\Delta=1/(2\beta)$ ($\beta=0.5$ → 1.0, $\beta=0.25$ → 2.0).
- Reference values enter only through $\Delta$: shifting both refs by the same $c$ leaves $\Delta$ and the loss unchanged; swapping the preferred/rejected sides together with their refs leaves $\Delta$ — and thus the loss — unchanged.
- Invalid input raises `ValueError`: non-tensor / non-1-D / non-float / shape or dtype mismatch / empty batch / NaN·Inf / $\beta\le 0$.
- CPU, deterministic, no random number calls.
