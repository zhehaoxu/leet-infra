# PEFT: Adapter Bottleneck Feed-Forward with Residual

## Problem

**Adapter** (Houlsby et al., arXiv:1902.00751) inserts a small "bottleneck-expand" feed-forward module between frozen Transformer layers: first downsample the \(d\)-dimensional input to \(r\) (\(r\ll d\)), apply a nonlinearity, then project back up to \(d\), and finally **add the original input back** (residual). During fine-tuning only these small side modules are trained; the backbone is entirely frozen.

The forward pass of a single Adapter:

\[y = W_\text{up}\,\sigma\!\bigl(W_\text{down}\,x\bigr) + x\]

\(\sigma\) is ReLU. The reduction ratio \(d/r\) is exactly where "parameter efficiency" comes from, and the residual guarantees that the insertion is approximately the identity at the start.

## Rules

Given input `x` of shape \((\dots, d)\), down-projection `W_down` of shape \((r, d)\), up-projection `W_up` of shape \((d, r)\):

1. Down-projection \(h = x\,W_\text{down}^\top\) (\((\dots, r)\)).
2. Nonlinearity \(a = \operatorname{ReLU}(h)\).
3. Up-projection \(u = a\,W_\text{up}^\top\) (\((\dots, d)\)).
4. Residual \(y = u + x\).

Return `y` (shape \((\dots, d)\)). float32 throughout, using a ready-made PEFT library is forbidden; for grading determinism, Dropout is **not** applied.

## Examples

`x` (2×3)\(=\begin{bmatrix}0.61&1.62&-0.80\\ 0.52&-0.21&-0.07\end{bmatrix}\), `W_down` (2×3)\(=\begin{bmatrix}0.4&-0.2&0.1\\ 0.0&0.3&-0.5\end{bmatrix}\), `W_up` (3×2)\(=\begin{bmatrix}0.2&0.5\\ -0.3&0.1\\ 0.4&-0.2\end{bmatrix}\) (\(d=3,r=2\)):

```
y = [[ 1.053,  1.709, -0.977],
     [ 0.569, -0.283,  0.027]]
```

## Constraints

- CPU float32; `x` supports 2D and 3D (last dim is \(d\)).
- The residual requires \(W_\text{up}\)'s output dim to equal the input dim \(d\), hence `W_up` shape \((d,r)\).
- With `W_up=0` the up-projection is always 0 and \(y=x\) (a pure residual bypass).
