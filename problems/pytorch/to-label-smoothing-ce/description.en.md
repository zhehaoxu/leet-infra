# Label Smoothing: Cross-Entropy with Label Smoothing

## Problem

A one-hot target pushes all the probability mass onto the correct class, which drives the model to output **over-confident** logits (softmax saturation, vanishing gradients, worse calibration, a tendency to overfit). **Label Smoothing** (Szegedy et al. 2016, R-SLU) replaces the hard label $y$ with a "soft label": keep $1-\alpha$ for the correct class and spread $\alpha$ uniformly over all $V$ classes. The cross-entropy $\mathcal{L}=-\sum_v q_v\log p_v$ ($q$ is the target distribution, $p$ the predicted softmax) then splits into two terms:

$$\mathcal{L}_\text{LS} = (1-\alpha)\underbrace{\big(-\log p_{y}\big)}_{\text{standard NLL}} + \alpha\underbrace{\Big(-\tfrac{1}{V}\sum_v \log p_v\Big)}_{\text{regularizer toward the uniform distribution}}$$

The second term equals $\alpha\cdot\mathrm{KL}(\text{uniform}\,\|\,p)$ plus a constant — it gently nudges the prediction toward the uniform distribution and suppresses over-confidence. This is exactly its connection to `kl_div`.

## Rules

Given `logits` $(N, V)$, `target` $(N,)$ (the correct-class index for each row), and the scalar `alpha` $\alpha$:

1. $\log p = $ `log_softmax(logits, dim=-1)` (numerically stable, subtract the maximum first)
2. NLL term: $u_n = -\log p[n, $`target[n]`$]$
3. Smoothing term: $s_n = -\frac{1}{V}\sum_v \log p[n, v]$ (the mean of $-\log p$ over all classes)
4. Per-sample: $(1-\alpha)\,u_n + \alpha\, s_n$
5. Take the mean over the batch

Return a 0-dim scalar `torch.Tensor` = the mean label-smoothing loss. It must reduce exactly to standard cross-entropy when $\alpha=0$.

## Example

```
alpha = 0.1, V = 3
logits = [[2.0,1.0,0.1],[0.5,2.5,1.0],[1.0,1.0,3.0]]
target = [0, 1, 2]

After log_softmax, row by row:
  row0  logp≈[-0.7867,-1.7867,-2.6867]  NLL=0.7867  smoothed mean=(0.7867+1.7867+2.6867)/3=1.7534
  row1  logp≈[-2.3133,-0.3133,-1.8133]  NLL=0.3133  smoothed mean=1.4800
  row2  logp≈[-2.4076,-2.4076,-0.4076]  NLL=0.4076  smoothed mean=1.7409
Per-sample 0.9*NLL + 0.1*smoothed = [0.8834, 0.4299, 0.5409]
mean = 0.436532
(when alpha=0, mean = 0.320977, equal to nn.functional.cross_entropy)
```

## Constraints

- CPU float32; $\alpha \in [0,1)$; `target` is a valid class index.
- Must use a stable log-softmax (`log(softmax)` without subtracting the maximum overflows).
- With $\alpha=0$ it is bitwise identical to `F.cross_entropy(..., reduction='mean')`; return a 0-dim scalar tensor.
