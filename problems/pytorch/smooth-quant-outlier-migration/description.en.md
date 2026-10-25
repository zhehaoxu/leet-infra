# SmoothQuant: Activation Outlier Channel Migration

## Problem

The biggest obstacle to W8A8 quantization is the **outlier channels** in the **activations**: the magnitudes of a few feature channels can be orders of magnitude larger than the rest; per-tensor quantization has its range stretched by these outliers, squashing all remaining channels to 0. The key insight of **SmoothQuant** (Xiao et al., 2023, [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)) is that activations are hard to quantize while weights are easy, so it uses a per-input-channel **mathematically equivalent scaling factor** $s_j$ to "migrate" part of the activation outliers onto the weights — divide $X$ by $s$ and multiply $W$ by $s$; mathematically $XW^\top$ stays exactly the same, yet the dynamic ranges on both sides are "smoothed out" and become easier to quantize.

## Rules

Given activations $X$ of shape $(T, C)$ ($T$ tokens, $C$ input feature channels), weights $W$ of shape $(O, C)$, and a balance coefficient $\alpha \in [0,1]$. For each input channel $j$:

$$s_j = \frac{\left(\max_i |X_{ij}|\right)^{\alpha}}{\left(\max_o |W_{oj}|\right)^{1-\alpha}}$$

Then migrate:

$$\hat{X}_{ij} = \frac{X_{ij}}{s_j},\qquad \hat{W}_{oj} = W_{oj}\cdot s_j$$

Return the triple `(s, X_hat, W_hat)` with shapes $(C,)$, $(T,C)$, $(O,C)$ respectively.

- Larger $\alpha$ → more of the outlier magnitude is left to the weights (which tolerate pressure better); $\alpha=0$ puts it all on the activations, $\alpha=1$ pushes it all to the weights.
- If a channel computes $s_j=0$ (that column's activations and weights are all 0), set $s_j=1$ to avoid division by zero.

## Examples

```
X = [[1, 2, 100],        W = [[0.1, 5.0, 0.2],
     [3, 0.5, 4],              [1.0, 0.3, 6.0]]
     [-2, 1, 8]],  alpha = 0.5

Per-channel absmax:
  X: col0=3, col1=2, col2=100
  W: col0=1, col1=5, col2=6
s = [sqrt(3)/sqrt(1), sqrt(2)/sqrt(5), sqrt(100)/sqrt(6)]
  = [1.732051, 0.632456, 4.082483]

X_hat = [[0.577350, 3.162278, 24.494898],
         [1.732051, 0.790569,  0.979796],
         [-1.154701, 1.581139,  1.959592]]
W_hat = [[0.173205, 3.162278, 0.816497],
         [1.732051, 0.189737, 24.494896]]
```

(The activations of outlier channel col2 are shrunk by $1/4.08$ while its weights are amplified by $\times 4.08$ — migration complete.)

## Constraints

- CPU float32; `s`, $\hat X$, $\hat W$ are all compared with `torch.allclose(atol=1e-4)`.
- The last dimension of $X$ and $W$ (the number of input channels $C$) must match.
- absmax is taken along "all the other dimensions": for $X$ along the token dim (dim=0), for $W$ along the output dim (dim=0).
