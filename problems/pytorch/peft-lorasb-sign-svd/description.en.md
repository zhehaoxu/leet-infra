# PEFT: LoRA-SB's Sign-Estimated Gradient SVD Initialization

## Problem Description

**LoRA-SB** (arXiv:2411.19557, CERT-Lab) replaces heuristic LoRA initialization with a
data-driven one: run one forward/backward pass on a small sample (~170 examples),
compress the accumulated gradient into its **sign estimate**
\(\widehat{\Delta W}=-\mathrm{lr}\cdot\mathrm{sign}(\nabla W)\) (the most aggressive
first-order compression: direction only, magnitudes all set to lr), then take a
**randomized low-rank SVD** of that estimate and split the factors ASYMMETRICALLY
into the LoRA sides — \(B=V\), \(A=(U\,\mathrm{diag}(s))^\top\), so \(B@A\)
reconstructs the sign update. The r_squared mode further inserts an \(r\times r\)
latent mapping \(M\) (initialized to \(\mathrm{diag}(s)\)) between A and B and
trains ONLY M; the SVD factors are frozen.

This problem replicates the deterministic accounts of the official implementation
(CERT-Lab/LoRA-SB):

- **sign update** (`gradient_utils.py`): \(\widehat{\Delta W}=-\mathrm{lr}\cdot\mathrm{sign}(g)\),
  `sign(0)=0`; every nonzero magnitude collapses to \(\pm\mathrm{lr}\).
- **svd_grad** (`svd_utils.py::get_svd_grad`): the official caller passes
  \(X=\widehat{\Delta W}^\top\) ((in,out)); float32 pipeline + `manual_seed(seed)`;
  **a dummy `randn(n_cols, rank)` is drawn BEFORE `svd_lowrank` and never used** —
  the value is discarded but it **advances the RNG stream** from which svd_lowrank's
  internal random projection draws; removing that line changes the result (quirk);
  returns \(enc=U\,\mathrm{diag}(s)\ (d_\text{in},r)\), \(dec=V^\top\ (r,d_\text{out})\), \(s\),
  promoted to float64 at the boundary.
- **init_ab** (`initialization_utils.py`): \(B_w=dec^\top\ (d_\text{out},r)\),
  \(A_w=enc^\top\ (r,d_\text{in})\); `half_init_dec` re-inits the LOWER \(\lceil r/2\rceil\)
  rows of dec (PRE-transpose) with `kaiming_uniform_(a=√5)`
  (bound \(=1/\sqrt{d_\text{out}}\); fan_in looks at the initialized matrix's columns);
  `random_init` re-inits enc THEN dec fully on the same global stream (kaiming a=√5;
  enc (in,r) has fan_in=r); half init runs first, its values are overwritten by
  random_init, but the **RNG-stream footprint survives** (different draw counts →
  different final values).
- **latent accounts** (`latent_utils.py`): the r_squared forward is
  \(B(M(A(x)))\cdot\text{scaling}\) (three chained linears, \(M\) a general asymmetric
  matrix), mathematically equal to the merged \(\Delta W=B@M@A\cdot\text{scaling}\);
  grad mode sets \(M=\mathrm{diag}(s)\).
- **non-r_squared fallback**: \(A\sim\mathcal N(0,10^{-5})\) near-zero init, only B trains.

## Task

- `mode="sign_update"`: grad (out,in) and lr in, return \(-\mathrm{lr}\cdot\mathrm{sign}(grad)\).
- `mode="svd_grad"`: X (in,out), rank, niter, seed in, return (enc, dec, s) (float64).
- `mode="svd_grad_nodummy"`: same but WITHOUT the dummy RNG draw (control path,
  not the official flow).
- `mode="init_ab"`: dec, enc and half_init_dec / random_init / seed in, return (a_w, b_w).
- `mode="latent_forward"` / `mode="delta_weight"`: as in the formulas above.
- `mode="latent_init"`: s in, return \(\mathrm{diag}(s)\).
- `mode="a_near_zero"`: r, in_features, seed in, normal_(0, 1e-5) RNG replay account.
- All floats in/out as float64; the SVD pipeline follows the official float32 internally.

## Example

\(d_\text{out}=6, d_\text{in}=8, r=3\), grad = randn(6,8) (seed 0), lr=0.01:

```
sign_update(grad, 0.01) -> (6,8), nonzero = ∓0.01, zeros = 0
svd_grad(X, 3, niter=10, seed=42) -> enc:(8,3), dec:(3,6), s:(3,)
  s = [0.04752158001065254, 0.02828427217900753, 0.028284268453717232] (bit-anchored float32 pipeline)
init_ab(dec, enc, random_init=True, seed=7) -> a_w:(3,8), b_w:(6,4)
latent_forward(x, a_w, M, b_w, 2.0) == x @ delta_weight(b_w, M, a_w, 2.0).T
```

With the same seed, the s of `svd_grad` and `svd_grad_nodummy` are **not equal** —
hard proof that the dummy draw advances the RNG stream.

## Constraints

- `svd_lowrank` uses the official q=rank and niter arguments; rank<=0 raises ValueError.
- `kaiming_uniform_` takes no generator argument; replay via the official global
  stream + manual_seed.
- Parts where M, A, B are explicit inputs do not recompute the SVD.
