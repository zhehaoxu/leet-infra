# Quantization: Searching SmoothQuant's α by W8A8 Output Error

## Problem

SmoothQuant migrates activation outliers into the weights with a per-input-channel scale \(s_j\): divide \(X\) by \(s\), multiply \(W\) by \(s\), and \(XW^\top\) is unchanged mathematically. How much to migrate is controlled by one balance exponent \(\alpha\):

\[s_j=\frac{\left(\max_i|X_{ij}|\right)^{\alpha}}{\left(\max_o|W_{oj}|\right)^{1-\alpha}}.\]

At \(\alpha=0\) the activations absorb the outliers alone (\(s_j=1/\max|W_{\cdot j}|\)); at \(\alpha=1\) everything is pushed to the weights. **Which end is better cannot be answered by the formula**: it depends on the quantization error on each side, so one has to run the full W8A8 fake quantization and select by output error. Given calibration activations, weights and an \(\alpha\) grid, run quantize-dequantize for every \(\alpha\), compute the output MSE, and return the best \(\alpha\), its scales, and the whole error curve.

## Rules

For every \(\alpha\) in the grid:

1. **Scales**: per input channel, \(s_j=\text{sx}_j^{\alpha}/\text{sw}_j^{1-\alpha}\) with \(\text{sx}_j=\max_i|X_{ij}|\) and \(\text{sw}_j=\max_o|W_{oj}|\). If \(\text{sx}_j=0\) or \(\text{sw}_j=0\) (degenerate channel), set \(s_j=1\).
2. **Migration**: \(X'=X/s\) (divide by column), \(W'=W\cdot s\) (multiply by column).
3. **Quantize activations (per token)**: each token row gets a symmetric scale from its own absmax, \(\text{scale}_i=\max_j|X'_{ij}|/q_{\max}\),
   \[
   q_{ij}=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{X'_{ij}}{\text{scale}_i}\right),-q_{\max},q_{\max}\right),\qquad \hat X_{ij}=q_{ij}\,\text{scale}_i.
   \]
   An all-zero row yields an all-zero \(\hat X\) (no NaN).
4. **Quantize weights (per output channel)**: same rule per output-channel row, \(\text{scale}_o=\max_j|W'_{oj}|/q_{\max}\), giving \(\hat W\).
5. **Error**: \(\text{MSE}=\frac{1}{T\cdot O}\sum_{i,o}\left[(XW^\top)_{io}-(\hat X\hat W^\top)_{io}\right]^2\), computed in float64.

Pick the \(\alpha\) with the smallest MSE; ties go to the **first** occurrence (the grid may contain duplicates).

Return a triple:

- `best_alpha`: float, the best \(\alpha\);
- `best_scales`: float32 \((C,)\), the \(s\) at the best \(\alpha\);
- `mses`: float32 \((\text{len(alphas)},)\), the output MSE of each \(\alpha\) in input order.

Here \(q_{\max}=2^{\text{bits}-1}-1\), and rounding follows `torch.round` (nearest, ties to even).

## Task

- Inputs: `X` float32 \((T,C)\), `W` float32 \((O,C)\), `alphas` (non-empty list of floats in \([0,1]\), duplicates allowed), `bits` an integer in 2..8.
- Output: `(best_alpha, best_scales, mses)` with `best_scales` and `mses` float32, tolerance `1e-5` (relative \(10^{-4}\) for MSE).
- Raise `ValueError` when `bits` is out of range; other inputs are guaranteed by the statement, no extra validation is required.

## Example

```
X = [[ 0.9, 1.1, 55.0],        W = [[ 0.4, -0.6,  0.05],
     [ 1.2, 0.7, 62.0],             [-0.5,  0.3, -0.04]]
     [-0.8, 1.4, 70.0]]
alphas = [0.0, 0.25, 0.5, 0.75, 1.0],  bits = 8
```

Per-channel absmax: `sx = [1.2, 1.4, 70.0]`, `sw = [0.5, 0.6, 0.05]`. Output MSE per α:

| α | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| MSE | 1.9476e-4 | 1.6137e-4 | **9.4523e-6** | 3.9517e-5 | 1.9066e-4 |

```
best_alpha  = 0.5
best_scales = [1.549193, 1.527525, 37.416573]   # sqrt(1.2/0.5), sqrt(1.4/0.6), sqrt(70/0.05)
```

(The outlier channel 2 is not squeezed enough at \(\alpha=0\), and the weights are pushed too hard at \(\alpha=1\); the middle value wins.)

## Constraints

- Quantization is **symmetric**: code range \([-q_{\max},q_{\max}]\), not \([-q_{\max}-1,q_{\max}]\).
- Activations get a scale per **token** (row), weights per **output channel** (row); both use their own row absmax, never a per-tensor scale.
- Compute \(s_j\) in float64; channels with `sx` or `sw` equal to 0 use \(s_j=1\) to avoid division by zero and `inf`.
- Only the returned values are judged; any implementation style is fine.
