# Optimizer: AdamP Scale-Invariant Direction Projection (Cross-Parameter Wedge)

## Problem

Weight decay is redundant along **scale-invariant** directions such as normalization layers (BN/LayerNorm): scaling \(w\) up or down along its own direction does not change the function, so decay along that direction has no regularizing effect and instead interferes with the adaptive update, which tends to cause "overfitting-like oscillation" late in training. AdamP (arXiv:2009.00329, naver) **removes the component of the Adam update vector that is parallel to the parameter (the radial component)**, keeping only the tangential component that genuinely changes the direction, and applies weaker weight decay to scale-invariant parameters (multiplied by `wd_ratio`).

This problem asks you to **subclass `torch.optim.Optimizer`** and implement AdamP, updating parameters in place via `step()`; the judge feeds a fixed gradient sequence and compares the final parameters against an independent reference element by element.

## Rules

- Implement the factory function `solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, delta=0.1, wd_ratio=0.1)`, returning an instance of a `torch.optim.Optimizer` subclass; `params` is a list of tensors to optimize and `step()` must **update those very tensors in place**. Fix `nesterov=False`, `cautious=False`, `weight_decouple=True`.
- For each parameter \(p\) with gradient \(g\), states \(m, v\) (initialized to zero), at step \(t\) (incremented only when that parameter has a gradient):
  1. First/second moments: \(m \leftarrow \beta_1 m + (1-\beta_1)g\); \(v \leftarrow \beta_2 v + (1-\beta_2)g^2\)
  2. Set \(\sqrt{\hat\beta_2^{(t)}}=\sqrt{1-\beta_2^{\,t}}\) and \(\text{step}=\dfrac{lr}{1-\beta_1^{\,t}}\); the update vector is
     \[ \text{perturb} = \frac{\sqrt{\hat\beta_2^{(t)}}}{\sqrt{v}+\epsilon}\odot m \]
  3. **Projection (only when `p.dim() > 1`)**: start with \(r=1.0\) (the decay ratio), then test the per-channel view first and the whole-layer view second; if a test hits, remove the radial component, set \(r=\texttt{wd\_ratio}\) and stop immediately (details below).
  4. **Decoupled weight decay** (only when `weight_decay`≠0): \(p \leftarrow p\,(1 - lr\cdot\text{wd}\cdot r)\)
  5. Update: \(p \leftarrow p - \text{step}\cdot \text{perturb}\)
- **Projection test and removal**: flatten the tensor with \(v(\cdot)\) into two-dimensional views (channel view: \((C, \prod_{i>0}d_i)\); layer view: \((1, \text{numel})\)). Compute \(\big|\cos(g, p)\big|\) row by row per view, with a floor of \(\epsilon\) on the denominator. If
  \[ \max|\cos| < \frac{\texttt{delta}}{\sqrt{\text{num cols of the view}}} \]
  the group is deemed scale-invariant: let \(\hat n = p / (\|p\|_{\text{row}} + \epsilon)\) (normalized row-wise), sum the radial component \(\text{radial}=(\hat n\odot \text{perturb})\) over each row, then
  \[ \text{perturb} \leftarrow \text{perturb} - \hat n \cdot \text{radial}, \quad r \leftarrow \texttt{wd\_ratio} \]
  Once a view hits, `break` and do not test the other view. If neither view hits, \(r\) stays 1.0 and perturb is unchanged.
- Everything is float32. Parameters with `grad is None` are skipped entirely. Calling an existing optimizer is **forbidden** (you may use `torch.nn.functional.cosine_similarity` or write the cosine yourself).

## Examples

Initialize a single 2×3 parameter \(p=\begin{bmatrix}0.5&-0.2&0.1\\0.3&0.4&-0.1\end{bmatrix}\) with `lr=0.1`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, `delta=0.1`, `wd_ratio=0.1`, and feed five consecutive gradients (each of the same \(2\times3\) shape):

```
g₁=[[ 0.20,-0.10, 0.05],[ 0.10, 0.30,-0.20]]  g₂=[[-0.15, 0.20, 0.10],[ 0.05,-0.10, 0.30]]
g₃=[[ 0.10, 0.05,-0.20],[-0.30, 0.15, 0.10]]  g₄=[[ 0.25,-0.05, 0.10],[ 0.10,-0.20,-0.15]]
g₅=[[-0.10, 0.30,-0.05],[ 0.20, 0.10, 0.20]]
after five steps p ≈ [[ 0.268631, -0.252074, -0.082198],
            [ 0.129582,  0.171029, -0.093757]]
```

## Constraints

- Input tensors have dtype float32; the output is updated in place and the dtype is unchanged
- One-dimensional and zero-dimensional parameters **are not projected** (\(r\) is always 1.0); two dimensions and above go through the projection test
- Must match the independent reference exactly, step by step (for the convention see notes.md)
