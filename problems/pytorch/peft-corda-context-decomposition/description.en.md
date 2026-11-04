# PEFT: CorDA Context-Oriented Decomposition Adaptation

## Problem

**CorDA** (arXiv:2406.05223, iboing/CorDA) injects the **context** of the downstream task into LoRA initialization: it collects the **covariance matrix** \(C\) of input activations over a few calibration samples, decomposes \(W\cdot C\) with an SVD oriented by that context, then multiplies \(C^{-1}\) back into \(V\) — the factorizing orientation is decided by the task data. The knowledge-preserved mode (KPA) keeps the **smallest** \(r\) singular components as the trainable adapter (world knowledge stays in the frozen residual to mitigate forgetting); the instruction-previewed mode (IPA, following PiSSA) keeps the **largest** \(r\).

This problem re-implements the deterministic account of the official `cordalib/decomposition.py`:

- **Damped inverse**: \(fix = C + I\cdot\mathrm{mean}(\mathrm{diag}(C))\cdot damp\); `damp` starts at 0.01 and doubles while \(\|fix\cdot fix^{-1}-I\|\ge 0.05\).
- **Oriented decomposition**: \(w = W\cdot fix\); after `linalg.svd(w)`, \(V \leftarrow (V^\top C^{-1})^\top\). **Context identity**: for the full decomposition \(U\,\mathrm{diag}(S)\,V^\top = W\) holds exactly (the covariance cancels through the inverse); for \(r<rank\) the remainder lives in the frozen residual.
- **Component selection**: `first_eigen=False` (KPA) takes \(U[:,-r:],\,S[-r:],\,V[:,-r:]\); `first_eigen=True` (IPA) takes the first \(r\).
- **Frozen residual**: \(R = W - U\,\mathrm{diag}(S)\,V^\top\).
- **sigma_fuse, three modes** (each satisfies \(A@B = U\,\mathrm{diag}(S)\,V^\top\)): `UV` (\(A=U\sqrt S\), \(B=V^\top\sqrt S\), column scaling on both); `U` (\(A=US\), \(B=V^\top\)); `V` (\(A=U\), \(B=V^\top S\)).
- **Forward**: \(y = A\,(B\,x) + R^\top x\); at init \(y = W^\top x\) holds exactly (zero-delta property).

## Task

- `mode="damped_inv"`: cov, damp (default 0.01), tol (default 0.05) -> (fix, cov_inv, damp).
- `mode="decompose"`: weight, cov, r, first_eigen -> (U, S, V, residual, fix, cov_inv, damp); raises ValueError for r<=0 or r>rank.
- `mode="fuse"`: U, S, V, fuse -> (A, B); raises ValueError for an unknown fuse mode.
- `mode="forward"`: x, a_w, b_w, residual -> y.
- The SVD pipeline runs in the official float32 (`.float()`); the damped inverse follows the input dtype.

## Example

\(d_\text{out}=6, d_\text{in}=5, r=2\), W = randn(6,5), X = randn(64,5) (seed 2), cov = \(X^\top X/64\):

```
damped_inv(cov) -> damp = 0.01 (well-conditioned covariance resolves in one step)
decompose(W, cov, 2) -> S = [1.2007561922073364, 0.7124348878860474] (float32 pipeline anchor)
  U[0,0] = 0.3145820200443268, V[0,0] = -0.2790605425834656
decompose(W, cov, 5) (full) -> U diag(S) V^T == W (rel err < 1e-5, float32 svd precision)
fuse(U, S, V, "UV") -> A=U*sqrt(S), B=V^T*sqrt(S); A@B == U diag(S) V^T
forward(x, A, B, R) == x @ W^T (zero delta)
```

## Constraints

- `2 <= r <= min(in, out)`; invalid r raises ValueError.
- The damp inside fix = cov + I*mean(diag)*damp is part of the return value (the starting damp is a parameter).
- KPA and IPA selections are mutually exclusive and both must reconstruct W exactly together with the frozen residual.
