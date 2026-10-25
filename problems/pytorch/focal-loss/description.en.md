# Focal Loss

## Problem

Standard cross-entropy treats every sample equally, but under extreme class imbalance (e.g. in object detection, where background far outnumbers foreground), a large number of "easy negative samples" dominate the gradient. RetinaNet (Lin et al., arXiv:1708.02002) proposes **Focal Loss**: multiply the cross-entropy by a modulating factor \((1-p_t)^\gamma\), suppressing the loss of samples that are "already predicted correctly (large \(p_t\))" and focusing on hard samples.

\[ \text{FL}(p_t) = -\,\alpha_t\,(1-p_t)^\gamma \log(p_t),\quad p_t = \text{softmax}(\text{logits})[\text{target}] \]

## Rules

Implement `solve(logits, target, gamma=2.0, alpha=None, reduction="mean")`. `logits` (float32, \((N, C)\)) are unnormalized class scores, and `target` (int64, \((N,)\)) are the ground-truth class indices. Let \(p_t\) be the softmax probability of the true class and \(\log p_t\) its logarithm; the per-sample loss is \(-(1-p_t)^\gamma \log p_t\); if `alpha` is not `None` (a scalar), multiply by `alpha` as well. Reduce according to `reduction` (`"mean"`/`"sum"`/`"none"`). An invalid `reduction` must raise `ValueError`.

## Examples

\(N=2, C=2\), `logits=[[2.0,0.5],[0.1,1.0]]`, `target=[0,1]`, `gamma=2.0, alpha=None`:

```
row 0: p_t=softmax[0]=0.81757 -> -(1-0.81757)^2*log(0.81757)=0.0067029
row 1: p_t=softmax[1]=0.71092 -> -(1-0.71092)^2*log(0.71092)=0.0285035
mean = 0.0176031645
```

## Constraints

- CPU float32; \(p_t\) is the softmax probability of the true class, and `log(p_t)` uses the numerically stable `log_softmax`.
- When `gamma=0` and `alpha=None`, Focal degenerates exactly to standard cross-entropy (it should match `cross_entropy`'s per-sample NLL).
- `alpha` is a scalar weight or `None`.
- Support the three `reduction` values; any other string raises `ValueError`.
