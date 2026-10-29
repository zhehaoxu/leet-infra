# RLOO: Leave-One-Out Baseline Advantage

## Problem

**RLOO** (REINFORCE Leave-One-Out, Ahmadian et al. 2024, *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*, arXiv:2402.14740, ACL 2024) is a simplification of the actor-critic setup PPO uses in RLHF: since each prompt already samples several answers, **the mean reward of the other answers in the group can serve as the baseline** — no value network to train.

For the $i$-th answer to the $b$-th prompt:

$$A_{b,i}=r_{b,i}-\frac{1}{G-1}\sum_{j\ne i}r_{b,j},$$

where $G\ge2$ is the group size. Two key properties:

- **Unbiased**: the baseline $\frac{1}{G-1}\sum_{j\ne i}r_{b,j}$ does not depend on the sampling of answer $i$, so its product with $\nabla_\theta\log\pi_\theta(y_i)$ has zero expectation and introduces no bias into the policy gradient;
- **Leave-one-out**: the denominator is $G-1$, not $G$, and the sum excludes the sample itself. Using the group mean including self makes $A_{b,i}$ negatively correlated with $r_{b,i}$ and systematically shrinks the advantage (more strongly for small $G$).

This problem only performs that advantage computation, from a batch of scalar rewards.

## Rules

Given a reward matrix $(B,G)$ with $B\ge1$, $G\ge2$ and finite float entries:

1. For each $(b,i)$, take the mean of the **other** $G-1$ rewards in the group: $\big(\sum_j r_{b,j}-r_{b,i}\big)/(G-1)$.
2. Advantage $A_{b,i}=r_{b,i}-$ that mean.
3. Return a $(B,G)$ **float64** tensor with the same shape as the input.

It follows immediately that $\sum_i A_{b,i}=0$: group advantages are always zero-sum.

## Examples

```
rewards = [[1.0, 0.0, 0.5],
           [2.0, 2.0, 0.0]]      G = 3

group 0: A = [1 - (0+0.5)/2, 0 - (1+0.5)/2, 0.5 - (1+0)/2]
           = [0.75, -0.75, 0.0]
group 1: A = [2 - (2+0)/2, 2 - (2+0)/2, 0 - (2+2)/2]
           = [1.0, 1.0, -2.0]
```

Each group's advantages sum to 0. With the denominator mistakenly written as $G$, group 0 would give $[0.5,-0.5,0]$ and group 1 $[0.667,0.667,-1.333]$ — uniformly shrunken and not proportional to the correct values.

For $G=2$ the leave-one-out mean is just the other answer, so the advantage degenerates to the raw difference: $r=[1,0]\Rightarrow A=[1,-1]$.

## Constraints

- Return a $(B,G)$ float64 tensor; float32 inputs are promoted internally.
- Zero-sum per group: $\sum_i A_{b,i}=0$ (to $10^{-12}$ in floating point).
- Identical rewards within a group give all-zero advantages; for $G=2$ the result is the raw difference.
- Invalid input raises `ValueError`: non-tensor / not 2-D / non-float / NaN·Inf / $B<1$ / $G<2$. For $G=1$ the leave-one-out baseline is undefined; degenerating to 0 is not allowed.
- CPU, deterministic, no random number calls.
