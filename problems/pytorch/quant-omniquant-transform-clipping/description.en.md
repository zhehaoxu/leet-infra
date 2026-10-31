# Quantization: OmniQuant's Equivalent Transform and Weight Clipping Joint Search

## Problem description

Quantizing one weight matrix to 4 bits is not served by "rotate/scale" alone or by "clip" alone: OmniQuant puts both knobs into **one search**. The first knob is the **channel-wise equivalent transform** - a learnable diagonal magnitude \(s\) plus a rank-one correction \(\mathbf 1\,\text{shift}^\top/n\), left-multiplied onto the weight and, through its approximate inverse, right-multiplied onto it, flattening the activation's dynamic range into the weight side. The second knob is **weight clipping** (WSC, the initialization of Learnable Weight Clipping) - the largest entries of the transformed weight are squeezed under a threshold, trading amplitude for usable code numbers.

Given the weight of one square linear layer, calibration activations that have **already been mean-shifted per column**, an alpha grid and a WSC candidate table, run for every \((\alpha,\text{choice})\) pair: "fix \(s\) -> fix shift -> assemble \(T\) and \(T_{\text{inv}}\) -> transform -> clip -> row-wise symmetric integerization -> layer reconstruction error", and return the winning pair, the transform matrices and clipped weight of that pair, the whole loss table, and the initial LWC bound.

## Rules

Let \(n\) be the side of the square, \(T\) the number of activation tokens, \(\mathbf 1\) the all-one column vector of length \(n\) and \(\delta\) the Kronecker symbol. The per-channel magnitude statistics are

\[
s1_i=\max_j|W_{ij}|\ (\text{per output channel}),\qquad s2_j=\max_t|X_{c,tj}|\ (\text{per input channel}).
\]

For every \(\alpha\) in `alphas` (outer loop, in input order):

1. **Mixed scale**: \(s=\alpha\,s1+(1-\alpha)\,s2\). Upstream reuses \(\beta\) as \(\alpha\), so
   \(\text{shift}=\alpha\,(s1-s)/s1\); wherever \(s1_i=0\) take \(\text{shift}_i=0\).
2. **The two transform matrices** (both \(n\times n\)):
   \[
   T=\operatorname{diag}(s)\,(I-\mathbf 1\,\text{shift}^\top/n),\qquad
   T_{\text{inv}}=\operatorname{diag}(1/s)\,(I+\mathbf 1\,\text{shift}^\top/n),
   \]
   i.e. elementwise \(T_{ij}=s_i(\delta_{ij}-\text{shift}_j/n)\) and \((T_{\text{inv}})_{ij}=(\delta_{ij}+\text{shift}_j/n)/s_i\).
   \(T\) left-multiplies and \(T_{\text{inv}}\) right-multiplies the weight: \(W'=T\,W\,T_{\text{inv}}\).
3. **Deviation from the identity**: \(\text{norm\_comp}=\lVert T\,T_{\text{inv}}-I\rVert_\infty=\max_{ij}|(T\,T_{\text{inv}}-I)_{ij}|\).
   Multiplying the two expressions gives \(T\,T_{\text{inv}}=I-\big(\sum_k\text{shift}_k/n^2\big)\mathbf 1\,\text{shift}^\top\),
   so the deviation is driven entirely by the shift sum and is exactly 0 at \(\alpha=0\) and \(\alpha=1\).
4. **Weight clipping (WSC)**: flatten `wsc_w` (the caller-supplied per-block \(\max|W|\), any shape), set
   \(w_2=\min_i w_i\) and \(w_3=\max_i|w_i|\), and map the elements of `wsc_choices` to scalar thresholds **in source order**:

   ```
   min -> w2      ma -> w3/4      ma2 -> w3/2      2 -> w2/8      1 -> w3/8
   ```

   Clipping acts on the whole matrix: \(W^c=\operatorname{clamp}(W',-t,t)\).
5. **Row-wise symmetric integerization**: \(q_{\max}=2^{\text{bits}-1}-1\), \(q_{\min}=-2^{\text{bits}-1}\);
   the scale of row \(r\) is that row's largest \(|W^c|\) divided by \(q_{\max}\) (an all-zero row uses scale 1 and code 0), and
   \(\text{deq}_{rj}=\operatorname{clamp}(\operatorname{round}(W^c_{rj}/\text{scale}_r),q_{\min},q_{\max})\cdot\text{scale}_r\).
6. **Layer reconstruction error**: the reference is the **transformed, unquantized** \(W'\), so quantization noise is the only penalty:
   \[
   \mathcal L(\alpha,\text{choice})=\frac{1}{T\cdot n}\sum_{t,j}\Big((X_cW'^\top)_{tj}-(X_c\,\text{deq}^\top)_{tj}\Big)^2 .
   \]

`losses` flattens \((\alpha,\text{choice})\) into one dimension: `losses[i * len(wsc_choices) + j]` corresponds to `(alphas[i], wsc_choices[j])`. Take the minimum; **ties go to the first occurrence** (compare the \(\alpha\) index first, then the choice index; only a strict improvement updates). All internal math is float64 and rounding is half-to-even; returned floating tensors are float32.

LWC is exercised only at its **initial value**: with \(\alpha_0=0.01\) (a constant fixed by the statement, not an input),
\(c=|w_{\text{sc},i}|\cdot 2\sigma(\alpha_0)\) is returned elementwise as `lwc_bound`, flattened like `wsc_w`.
Upstream writes \(\min(w,\,w\cdot 2\sigma(\alpha))\); at \(\alpha_0\) one has \(2\sigma(\alpha_0)>1\), so the \(\min\) does not bite - it only does after gradients push \(\alpha\) negative, and the iteration is out of scope here (hence `lwc_steps` changes no returned value).

## Task

- Inputs: `W` \((n,n)\), `Xc` \((T,n)\) (already mean-shifted per column; this problem does not re-center it), `alphas` (non-empty, elements in \([0,1]\)), `wsc_choices` (non-empty, elements among the five strings \(\{\text{min},\text{ma},\text{ma2},\text{2},\text{1}\}\)), `wsc_w` (any shape, non-empty), `n_bits` (2..16), `lwc_steps` (non-negative).
- Output: a dict with exactly the keys `best_alpha_index, best_alpha, best_wsc_choice, losses, s, shift_vec, transform_T, transform_T_inv, norm_comp, clipped_weight, lwc_bound`; floating tensors are float32 with tolerance \(10^{-5}\) (losses may use relative tolerance \(10^{-4}\)).
- Raise `ValueError` on: `W` not a 2-D square matrix, `W` and `Xc` disagreeing on the input-channel dimension, empty `alphas` or `wsc_choices`, an alpha outside \([0,1]\), a choice outside the five-element set, `n_bits` outside \([2,16]\), negative `lwc_steps`, empty `wsc_w`, or some \(s_i\) exactly 0 (its reciprocal is undefined).

## Example

```
W  = [[ 3.88,  1.75,  1.45, -0.85],        Xc = [[-0.27,  0.39,  0.96,  1.06],
     [-2.00,  1.35,  1.60,  1.30],               [ 0.12, -0.06, -0.09,  0.46],
     [ 0.88,  0.85, -0.35,  0.15],               [-0.03, -0.11, -0.84,  0.46],
     [ 1.75, -0.95,  1.10, -1.50]]               [ 0.10,  0.24,  0.31, -0.29],
                                                  [ 0.08, -0.46, -0.34, -1.69]]
alphas      = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
wsc_choices = ["min", "ma", "ma2", "2", "1"]
wsc_w       = [3.88, 1.75, 1.60, 1.50]
n_bits = 4,  lwc_steps = 0
```

Here \(s1=[3.88,2.00,0.88,1.75]\), \(s2=[0.27,0.46,0.96,1.69]\), \(w_2=1.50\), \(w_3=3.88\). Loss table (rows = \(\alpha\), columns = choice):

```
                 min         ma         ma2           2           1
a=0.00    6.86674e-01  1.10736e+00  7.63157e-01  2.26830e+00  1.71558e+00
a=0.15    6.76201e-02  2.67329e-01  9.47780e-02  1.32119e+00  7.73547e-01
a=0.20    4.80141e-02  2.64371e-01  3.53595e-02  1.35942e+00  7.97988e-01
a=0.25    4.17612e-02  2.71005e-01  2.82965e-02  1.44052e+00  8.50101e-01   <- best
a=0.30    5.31183e-02  3.29183e-01  2.95982e-02  1.55287e+00  9.30576e-01
a=0.35    9.69807e-02  3.90868e-01  2.85752e-02  1.69158e+00  1.03530e+00
a=0.40    1.34912e-01  4.78205e-01  4.18000e-02  1.85496e+00  1.16291e+00
```

```
best_alpha_index = 3      best_alpha = 0.25      best_wsc_choice = "ma2"
s         = [1.172500, 0.845000, 0.940000, 1.705000]
shift_vec = [0.174452, 0.144375, -0.017045, 0.006429]
norm_comp = 4.143121e-02
clipped_weight = [[ 1.940000,  1.940000,  1.623360, -0.577780],
                  [-1.437042,  1.290990,  1.319418,  0.643622],
                  [ 0.674053,  0.849912, -0.478703,  0.080629],
                  [ 1.940000, -1.940000,  1.767677, -1.505982]]
lwc_bound = [3.899400, 1.758750, 1.608000, 1.507500]
```

The threshold \(t=w_3/2=1.94\) shaves only the outlier input channel of magnitude \(3.88\) in column 0, while \(\alpha=0.25\) pulls the per-row absmaxes towards each other so the row scales shrink; \(\alpha=0\) (pure activation scale) and the wider/stricter clip choices above are all worse.

In the second example the weight is the identity and \(s1=s2\), so every \(\alpha\) gives \(T=I\), \(\text{shift}=0\) and identical losses across the three alphas - ties keep the first occurrence:

```
W = [[1.0, 0.0],   Xc = [[ 1.0, -1.0],
     [0.0, 1.0]]         [-1.0,  1.0]]
alphas = [0.0, 0.2, 0.4]
wsc_choices = ["min", "ma", "ma2", "2", "1"]
wsc_w = [1.0, 1.0],  n_bits = 4,  lwc_steps = 0
```

```
best_alpha_index = 0   best_alpha = 0.0   best_wsc_choice = "min"
losses = [0.0, 5.625e-01, 2.5e-01, 7.65625e-01, 7.65625e-01]   (same for all three alphas)
s = [1.0, 1.0]   shift_vec = [0.0, 0.0]   norm_comp = 0.0
clipped_weight = [[1.0, 0.0], [0.0, 1.0]]
lwc_bound = [1.004999995231628, 1.004999995231628]
```

## Constraints

- The transform only covers **square layers** (`in_features == out_features`): \(s1\) and \(s2\) must have the same length to be convex-combined, and only then are \(T\) and \(T_{\text{inv}}\) both \(n\times n\).
- `wsc_w` is an input, not something recomputed: every \(\alpha\) shares one per-block amplitude table, and `lwc_bound` depends only on it and on the constant \(2\sigma(\alpha_0)\).
- The threshold is a **scalar** (from \(w_2,w_3\) over the whole table), not a per-row absmax; the factor \(2\sigma(\alpha_0)=1.00499996\) must be visible through `lwc_bound`.
- The judge only inspects the returned dict; the implementation strategy is free.
