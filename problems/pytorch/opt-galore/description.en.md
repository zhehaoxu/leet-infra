# Optimizer: GaLore Low-Rank Projected-Gradient AdamW

## Problem

The first- and second-moment states of AdamW are as large as the tensor being optimized, which dominates memory. GaLore (Zhao et al., arXiv:2403.03507) observes that gradient matrices stay approximately low-rank, so it first projects the gradient into a **low-rank subspace**, runs the moment estimates and the adaptive update inside that small space, and finally maps the update back — shrinking the optimizer state from \(O(d)\) to \(O(r)\) with \(r\ll d\). The original paper refreshes the projection basis with a (randomized) SVD every `update_proj_gap` steps; to keep this problem judgeable on CPU and free of all randomness, the basis \(P\) here is an **explicit per-parameter input**, and only the project → small-space Adam → map-back pipeline is graded, plus a **basis-recompute step ledger** (a pure integer schedule).

You must **subclass `torch.optim.Optimizer`** and implement this optimizer; `step()` updates the parameters in place. The judge builds small parameters (≤ 8×8), feeds a fixed gradient sequence through repeated `step()` calls, compares the parameter trajectory against an independent recomputation, and checks the recompute ledger of every projected parameter.

## Rules

- Factory function `solve(params, proj, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, update_proj_gap=200)`, returning an instance of a `torch.optim.Optimizer` subclass:
  - `params`: list of tensors to optimize (float32, 1-D or 2-D);
  - `proj`: a list of the same length as `params`; each entry is either `None` (the parameter is not projected and follows plain full-space AdamW) or a float32 2-D tensor \(P\) of shape `[p.numel(), r]` (\(r\ge 0\)). \(P\) is assumed to have orthonormal columns, but the implementation **must not rely on that, must not re-orthonormalize and must not run an SVD**;
  - `update_proj_gap`: an integer ≥ 1 that only affects the ledger and **never the numerical trajectory**.
- For every parameter \(p\) with a gradient at step \(t\) (1-based, counted per parameter independently), in this exact order:
  1. **State initialization on first sight**: \(m=0\), \(v=0\), sized to the projected-space dimension \(r\) (for unprojected parameters, \(r=\texttt{p.numel()}\), i.e. full space).
  2. **Ledger** (projected parameters only): if \((t-1)\bmod \text{update\_proj\_gap}=0\) (the first step \(t=1\) builds the basis, then every gap steps), append \(t\) to `state[p]["recompute_steps"]` (ascending, no duplicates). Also expose a pure method `recompute_at(t)` on the optimizer instance returning this boolean schedule.
  3. **Decoupled weight decay** (only if `weight_decay`≠0, before the moment estimates, in full space): \(p\leftarrow p\,(1-lr\cdot\text{wd})\).
  4. **Projection** (flatten row-major): \(\tilde g = P^\top g \in \mathbb{R}^{r}\).
  5. **Moments in the projected space**: \(m \leftarrow \beta_1 m + (1-\beta_1)\tilde g\); \(v \leftarrow \beta_2 v + (1-\beta_2)\,\tilde g\odot\tilde g\).
  6. **Bias correction and direction** (both inside \(\mathbb{R}^r\)): \(bc_1 = 1-\beta_1^{\,t}\), \(bc_2 = 1-\beta_2^{\,t}\), \(\hat u = \dfrac{m/bc_1}{\sqrt{v}/\sqrt{bc_2} + \epsilon}\).
  7. **Map back and apply** (in place): \(\Delta = \operatorname{reshape}(P\hat u,\ \text{shape}(p))\); \(p \leftarrow p - lr\cdot\Delta\).
- With \(r=0\) the mapped-back \(\Delta\equiv 0\) (only weight decay survives); a parameter with `grad is None` is skipped entirely (no weight decay, no ledger entry). Everything is float32; calling ready-made optimizers is forbidden.

## Example

Single parameter \(p=\begin{bmatrix}1.0 & -2.0\\ 0.5 & 0.0\end{bmatrix}\) (flattened length \(d=4\)) with the column-orthonormal basis \(P=\tfrac{1}{\sqrt2}\begin{bmatrix}1&0\\ 0&1\\ 1&0\\ 0&-1\end{bmatrix}\), `lr=0.1`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, `update_proj_gap=2`, fed three steps of gradients:

```
g₁ = [[ 0.10, -0.20], [ 0.30,  0.10]]
g₂ = [[-0.40,  0.05], [ 0.20, -0.10]]
g₃ = [[ 0.15, -0.10], [-0.25,  0.30]]
p after three steps ≈ [[0.903279, -1.860648], [0.404777, -0.133358]]
ledger recompute_steps = [1, 3]   # (t−1) mod 2 == 0: t=1 builds the basis, t=3 starts the next cycle
```

## Constraints

- Parameters ≤ 8×8, \(0\le r\le p.numel()\), plain float32 on CPU
- The moments and the bias correction live in the projected space while the update is applied in full space — do not mix in full-space moments
- The trajectory must match the independent recomputation element-wise (`atol=1e-5, rtol=1e-5`); the ledger is an integer sequence that must be exactly equal; with \(P=I\) the result must coincide with native AdamW
