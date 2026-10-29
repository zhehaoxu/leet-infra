# Diffusion: Frequency-Decoupled Guidance (FDG)

## Problem Statement

CFG applies a single guidance scale to the whole latent, but **low and high frequencies want opposite things**: push the low band (color blocks, composition) too hard and you lose diversity; push the high band (detail, texture) too little and the result looks soft. **Frequency-Decoupled Guidance** (FDG) first splits both predictions into frequency bands, runs **a separate CFG in each band**, and then reconstructs the data-space prediction:

\[ \text{pred}=\sum_l \text{pred}_l,\qquad \text{pred}_l=u_l+s_l\,(c_l-u_l) \]

Each band has its own scale \(s_l\) (the paper recommends larger values for high frequencies and smaller ones for low frequencies, e.g. 10 and 5 for SDXL). FDG also ships two optional knobs:

- **Parallel-component weighting** \(w_l\): split the band's difference \(c_l-u_l\) into components parallel and orthogonal to the conditional prediction \(c_l\), scale the parallel part by \(w_l\), and add the orthogonal part back. \(w_l=1\) is standard CFG; \(w_l=0\) keeps only the orthogonal direction — "change direction, not magnitude".
- **Per-level rescale** \(\varphi_l\): pull the band's guided result back to the conditional prediction's scale via the ratio of standard deviations, then mix it with the unscaled result by \((1-\varphi_l,\varphi_l)\). This damps the over-exposure seen at high guidance scales.

This problem treats the band decomposition/reconstruction operator (the Laplacian pyramid's `pyrup` upsampling-blur) as an external premise: the inputs are **already-split, equal-shape bands**, so the inverse transform reduces to summing the levels.

## Rules

Inputs are two equal-length tensor lists `cond_levels` / `uncond_levels`, ordered high to low frequency, each level shaped `(batch, ...)` with all levels sharing one shape; plus three optional lists `parallel_weights`, `level_rescale`, `active_levels` (defaults: all 1, all 0, all `True`).

For level \(l\):

- If the level is **disabled**: \(\text{pred}_l=c_l\).
- Otherwise:
  1. \(\delta_l=c_l-u_l\);
  2. if \(w_l\) is not close to \(1.0\) (`math.isclose` default tolerances):
     \[ \hat{c}_l=\frac{c_l}{\max(\lVert c_l\rVert_2,\,10^{-12})},\quad p=(\delta_l\cdot\hat c_l)\,\hat c_l,\quad o=\delta_l-p,\quad \delta_l=w_l\,p+o \]
     where the norm and inner product reduce over every dimension but the batch;
  3. \(\text{pred}_l=u_l+s_l\,\delta_l\);
  4. if \(\varphi_l>0\): let \(\sigma_{\text{ref}}\), \(\sigma_{\text{pred}}\) be the **unbiased** standard deviations (`torch.std` default) of \(c_l\) and \(\text{pred}_l\) over every dimension but the batch, then
     \[ \text{pred}_l=(1-\varphi_l)\,\text{pred}_l+\varphi_l\left(\text{pred}_l\cdot\frac{\sigma_{\text{ref}}}{\sigma_{\text{pred}}}\right) \]

Return the sum of all \(\text{pred}_l\) as a float32 tensor with the shape of a single band.

## Example

Two bands (high frequency first):

\[ c^{(0)}=\begin{bmatrix}1.0&-2.0&0.5\end{bmatrix},\quad u^{(0)}=\begin{bmatrix}0.5&-1.0&0.0\end{bmatrix},\quad c^{(1)}=\begin{bmatrix}0.2&0.4&-0.6\end{bmatrix},\quad u^{(1)}=\begin{bmatrix}0.1&0.2&-0.2\end{bmatrix} \]

with scales \(s=(10,\,5)\):

```
defaults            -> [[6.100000, -9.800000, 2.800000]]
parallel [0, 1]     -> [[0.861905, 0.676190, 0.180952]]
rescale 0.5 both    -> [[3.608426, -5.666953, 1.507397]]
```

The first row is just per-band CFG summed; the second drops the high band's parallel component entirely, keeping only the direction orthogonal to \(c^{(0)}\); the third pulls each band halfway back to its conditional prediction's scale.

## Constraints

- Both lists are non-empty and equal-length, each level has at least two dimensions (dim 0 is the batch), and all levels share one shape; cond and uncond of the same level must match too.
- Each entry of `guidance_scales` and `parallel_weights` is a finite non-negative real; each entry of `level_rescale` lies in \([0,1]\); each entry of `active_levels` must be a `bool` (no `0/1`). Any length mismatch raises `ValueError`.
- **Degenerate case**: when all levels share one scale with \(w_l\equiv1\) and \(\varphi_l\equiv0\), the output must equal a single CFG on the summed levels — that is, \(\sum_l u_l+s\,(\sum_l c_l-\sum_l u_l)\).
- With \(\varphi_l>0\), the level's prediction must not be constant per sample (otherwise \(\sigma_{\text{pred}}=0\) and the rescale is undefined); raise `ValueError` instead of producing `NaN`.
- A disabled level passes through **its own** conditional level, not some neighbouring level's intermediate result.
- Compute in float64 internally if you like, but return float32.
