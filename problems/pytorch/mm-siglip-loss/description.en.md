# Multimodal: SigLIP Sigmoid Contrastive Loss

## Problem

SigLIP (Zhai et al., *Sigmoid Loss for Language-Image Pre-Training*, arXiv:2303.15343) replaces CLIP's softmax normalization with a **pairwise sigmoid**: each (image \(i\), text \(j\)) pair is independently judged "positive/negative", with no need to gather the whole batch across GPUs for global normalization, so it is stable for small batches and easy to scale to very large batches.

For the similarity matrix \(s_{ij}=x_i^\top y_j\), first apply the affine transform \(z_{ij}=t\,s_{ij}+b\) (\(t=\) `logit_scale`, \(b=\) `logit_bias`), with labels \(y_{ij}=+1\) (diagonal positive pairs) and \(-1\) otherwise:

\[ \mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}\log\sigma\bigl(y_{ij}\,z_{ij}\bigr) \]

Note that the denominator is the batch size \(B\) (sum over all \(B^2\) pairs, then divide by \(B\)), not \(B^2\).

## Rules

Implement `solve(sim, logit_scale, logit_bias)`. Inputs: `sim` (float32, square matrix \((B,B)\), already the image×text dot products), and the scalars `logit_scale` and `logit_bias`. Return a 0-dimensional float32 loss scalar according to the formula above. `logsigmoid` must be numerically stable. Calling contrastive-learning libraries is forbidden.

## Examples

`sim=[[1.0,0.2],[0.3,0.9]]`, `logit_scale=5.0`, `logit_bias=-1.0`:

```
z = 5*sim - 1 = [[4.0, 0.0],[0.5, 3.5]]
y = [[+1,-1],[-1,+1]]
loss = -(1/2)*Σ log σ(y⊙z) = 0.857562
```

## Constraints

- CPU float32; `sim` is a \(B\times B\) square matrix whose diagonal holds the positive samples.
- The key difference from CLIP: pairwise sigmoid with no softmax global normalization, and the denominator is \(B\).
