# Quantization: AdaRound's Reconstruction-Error Candidate Search

## Task

Rounding a weight up or down is not always best done by nearest integer: nudging one weight up can be cancelled by the rest of its row. AdaRound (Nagel et al., ICML 2020, arXiv:2004.10501) turns this into a **reconstruction-error search** - give every weight a "round-up probability", then search which way that probability should move using the output error on calibration activations.

This problem gives a small weight matrix, a small **explicit** calibration matrix `X` and a per-tensor step `scale`. Implement the deterministic pipeline below: the soft initialisation, the **greedy per-element candidate search** and the hard-decode finalisation. No gradients, no backward pass, no sampling: the same input must produce a bit-for-bit identical answer.

## Rules

Let resolution = 2**num_iters, let n be the number of columns of `W`, z = W/scale - q the normalized residual, K0 = -0.1, K1 = 1.1, eps = 1e-6.

1. **Grid and residual**: q = floor(W / scale), r = W - q * scale in [0, scale), z = r / scale in [0, 1).
2. **Soft start**: h = clamp((z - K0) / (K1 - K0), 0, 1), then the **hard override** q = 0 => h = 1 (whenever the lower grid point is exactly 0, h is pinned to 1 regardless of the residual).
3. **Soft decode**: bar_h = clamp(floor(h * resolution) / resolution, eps, 1 - eps), and
   \[
   \alpha=\operatorname{sigmoid}\!\left(\frac{\operatorname{logit}(\bar h)+\log(\text{resolution})}{2}\right),\qquad
   \operatorname{logit}(u)=\log u-\log(1-u).
   \]
   Clamping to [eps, 1-eps] keeps the logit finite when bar_h is 0 or 1; both the +log(resolution) shift and the divide-by-2 temperature are copied from upstream.
4. **Objective**: at round i (i = 0..T-1, T = steps) use
   \[
   \mathcal L_i=\frac1n\sum_{o}\bigl\|X\,\bigl((q+\alpha)_{o}- (W/\text{scale})_{o}\bigr)\bigr\|^2\cdot\text{scale}^2
   \;+\;\beta_i\cdot\frac1{On}\sum_{o,j}\bigl|\alpha_{oj}-0.5\bigr|^{p_i},
   \]
   with p_i = 1 + (i + 1) / T and beta_i = p_i / log(lam). The first term is exactly the reconstruction error sum_o \|\|X(w_hat_o - w_o)\|\|^2 / n with w_hat = (q + alpha) * scale.
5. **Greedy per-element search**: every round visits the **not yet frozen** elements in **row-major order** (row 0 column 0, then row 0 column 1, ...). For each element try the 5 candidate offsets in this **fixed order**:
   \[
   \bigl[\,0,\ +1/\text{resolution},\ -1/\text{resolution},\ +1,\ -1\,\bigr].
   \]
   Each candidate replaces h_k with clamp(h_k + off, 0, 1), keeps every other element at its current value, re-decodes and evaluates \(\mathcal L_i\) over the **whole** `W`; take the candidate with the smallest objective. Update only on a strict improvement, so **ties keep the earlier candidate in the list** (e.g. after clamping, `0` and `+1` give the same trial h, and the winner is recorded as `0`).
6. **Commit and freeze**: the winner is committed as h_k <- clamp(h_k + off, 0, 1) and `chosen_offset[k]` stores that off (a later round overwrites the earlier value). After the commit, if |h_k - 0.5| > 0.5/resolution the element is **frozen** and never searched again; an element still inside that band is searched again next round, with the new offset **accumulated** on the already committed h_k.
7. **Per-round record**: after each round, evaluate \(\mathcal L_i\) once with the current alpha and store it in `soft_loss[i]`.
8. **Hard decode**: after all rounds, mask = (|h - 0.5| <= 0.5/resolution + eps). Elements inside the mask snap to the nearest grid point: alpha_hard = 1 if floor(h * resolution)/resolution >= 0.5 (exactly 0.5 rounds up), else 0; elements outside the mask use alpha_hard = round(alpha). Finally searched_weight = (q + alpha_hard) * scale and `searched_mse` = sum_o \|\|X(searched_o - w_o)\|\|^2 / n (**without** the regularizer).

All internal math is float64; returned float tensors carry `W`'s dtype (float32 or float64), `q` is int64 and `searched_mse` is a 0-dimensional tensor.

Return a dict with exactly 9 keys:

- `q` int64 (O, n): the lower grid point;
- `r` float (O, n): the absolute residual W - q * scale;
- `h` float (O, n): the softening variable after the search;
- `alpha` float (O, n): the soft round-up probability before the final hard decode;
- `betas` float (T,): the per-round regularizer weights beta_i;
- `soft_loss` float (T,): the objective L_i at the end of each round;
- `searched_weight` float (O, n): the hard-decoded quantized weight;
- `chosen_offset` float (O, n): the offset committed in the last round the element was searched (every element is searched once in the first round, so no element is untouched);
- `searched_mse` float 0-dim: the pure reconstruction error of the hard-decoded weight.

## Inputs and outputs

- Inputs: `W` float32/float64 (O, n), `X` same dtype (T_cal, n), `scale > 0`, `num_iters >= 1`, `steps >= 1`, `lam > 0` and `lam != 1`.
- Output: the 9-key dict above; tolerances `1e-5` for float32 and `1e-9` for float64.
- Raise `ValueError` for: `num_iters < 1`, `steps < 1`, `scale <= 0`, `lam <= 0` or `lam == 1`, `W` or `X` not 2-D, `W.shape[1] != X.shape[1]`, and `W`/`X` with different dtypes.

## Examples

Example 1: scale = 1.0, num_iters = 10 (resolution = 1024, fine step 1/1024 ~= 0.00097656), steps = 3, lam = 5.0, and `X` is the identity, so the reconstruction term collapses to "sum of per-element squared errors divided by the number of columns" and is hand-checkable. Numbers below come from float64 inputs.

```
W = [[ 0.34000,  0.90000,  1.55000],
     [ 0.99998, -0.72000,  2.00000]]
X = identity(3)

q               = [[0, 0, 1], [0, -1, 2]]
r               = [[0.34000, 0.90000, 0.55000], [0.99998, 0.28000, 0.00000]]
h               = [[0.00000000, 0.99902344, 0.54069010], [0.99902344, 0.00000000, 0.00000000]]
alpha           = [[0.03100777, 0.99902391, 0.97196826], [0.99902391, 0.03100777, 0.03100777]]
betas           = [0.82844658, 1.03555822, 1.24266987]
soft_loss       = [0.42640970, 0.41976967, 0.40138468]
searched_weight = [[0.00000, 1.00000, 2.00000], [1.00000, -1.00000, 2.00000]]
chosen_offset   = [[-1.00000000, -0.00097656, -0.00097656], [-0.00097656, -1.00000000, -1.00000000]]
searched_mse    = 0.13550000
```

(w = 2.0 sits exactly on the grid: r = 0, q = 2; candidate -1 drives h to 0, alpha drops to the 0.03 level and the final decode lands back on q + 0 = 2 with zero reconstruction error, while candidate `0` keeps alpha near 1 and would round up to 3. w = 0.34 has q = 0, so the h = 1 override fires, and candidate -1 still pulls it back to 0.)

Example 2: a single column, scale = 1.0, num_iters = 10, steps = 3, lam = 5.0, `X = [[20.0], [-10.0]]`. The first element has q = 0, so the h = 1 override fires; candidates `0`, +1/1024 and +1 all clamp to the same trial h = 1, their objectives are identical bit for bit, and the tie is broken toward the earliest candidate `0`.

```
W = [[0.99998], [0.05000]]
X = [[20.0], [-10.0]]

q               = [[0], [0]]
r               = [[0.99998], [0.05000]]
h               = [[1.00000000], [0.00000000]]
alpha           = [[0.99996875], [0.03100777]]
betas           = [0.82844658, 1.03555822, 1.24266987]
soft_loss       = [0.49565801, 0.49000997, 0.47233177]
searched_weight = [[1.00000], [0.00000]]
chosen_offset   = [[0.00000000], [-1.00000000]]
searched_mse    = 1.25000020
```

## Notes

- resolution depends only on `num_iters`: the fine candidate step is 1/resolution and both the freeze band and the final mask have half-width 0.5/resolution; `steps` only changes the p_i/beta_i schedule and the number of search rounds.
- The candidate order is exactly [0, +1/resolution, -1/resolution, +1, -1], the visit order is row-major and ties go to the earlier candidate; these three pins are what make the expected output unique.
- The objective is evaluated over the **whole** `W`: the reconstruction term sums over every output row and the regularizer averages over every element, not only the current element's row.
- Freezing depends only on whether the committed h leaves [0.5 - 0.5/resolution, 0.5 + 0.5/resolution], not on the objective value.
- Judging only looks at the returned dict; the implementation style is free.
