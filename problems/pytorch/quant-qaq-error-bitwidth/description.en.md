# Quantization: QAQ Error-Aware KV Bit-Width Solving

## Problem

Picking one global bit-width for the KV cache is a blanket decision: a (token, layer, head) unit with a small dynamic range wastes bits, while a wide one stays starved. QAQ's attention-aware idea (arXiv 2403.04643) goes the other way — for every quantization unit first **set an error budget** \(\varepsilon_{\max}\), then **solve for** the bit-width it needs, and finally run one affine quantize-dequantize with that width; outliers are picked out first and kept verbatim.

The inversion comes from an affine quantizer's rounding error. Spreading a unit's dynamic range \(\Delta\) over \(2^n\) grid points gives a step of \(\Delta/2^{n}\); rounding sits in \([-\Delta/2^{n+1},\ \Delta/2^{n+1}]\), whose uniform variance is \(\Delta^{2}/(12\cdot 4^{n})\). Requiring it below the target error \(\tau\) and solving for \(n\) is written in the implementation as

\[
n\ =\ \Bigl\lceil \log_2\Bigl(\frac{\Delta}{2\,\varepsilon_{\max}}+1\Bigr)\Bigr\rceil,\qquad
\Delta=\begin{cases}2\max|x|, & \text{symmetric}\\ \max x-\min x, & \text{asymmetric}\end{cases}
\]

both extremes are taken **over non-outlier elements only**. The error budget itself has two formulas (\(L=\) seq_len, \(\tau=\) target error):

\[
\varepsilon_{\max}^{\text{key}}=\sqrt{\frac{12}{\lVert q\rVert_2}\,\log\Bigl(\frac{L^{3}}{L-1}\,\tau^{2}+1\Bigr)},\qquad
\varepsilon_{\max}^{\text{value}}=\frac{\sqrt{12/L}\ \tau}{\max|A|}
\]

Here \(\lVert q\rVert_2\) is the query's 2-norm and \(\max|A|\) is the amax over the quantize dims of the last `last_n_attentions` rows of the attention. So the key branch's \(\varepsilon_{\max}\) depends only on \((L,\tau,\lVert q\rVert_2)\), not on the cache values, and is identical across units; only the value branch varies per unit.

Three conventions that are easy to get wrong: the mask uses \(\le\) on the low end and \(>\) on the high end; the bit-width is ceil'd then clamped, and only afterwards are the last `last_n-1` tokens unconditionally lifted to `n_bits_max`; and `method="normal"` swaps only the quantizer, never the minmax bit-width formula above.

## Task

Implement `solve(cache, attentions, key_or_value, level, symmetric, outliers_ratio, last_n_attentions, target_quantization_error, n_bits_min, n_bits_max, q_norm=None, method="uniform", quantile_upper=None, quantile_center=None)`:

- `cache`: float32/float64, `(n_batch, seq_len, n_layer, n_head, head_dim)`, i.e. the upstream post-permute layout;
- `attentions`: `None` for the key branch; `(n_layer, n_batch, n_head, q_len, seq_len)` for the value branch;
- `key_or_value`: `"key"` or `"value"`; `level`: `"token"/"layer"/"head"` mapping to quantize dims `(-3,-2,-1)/(-2,-1)/(-1,)`;
- `symmetric`: bool; `outliers_ratio`: in \([0,1)\); `last_n_attentions`: int `>= 1`; `target_quantization_error`: `> 0`;
- `n_bits_min`/`n_bits_max`: integers with `0 <= n_bits_min <= n_bits_max <= 16`; `q_norm`: a required positive number for the key branch;
- `method`: `"uniform"` or `"normal"`; `quantile_upper`/`quantile_center`: required for normal, entry \(n\) being a 1-D table of length \(2^{n}\).

Let \(k\) be the number of quantize dims and the unit shape \(\text{lead}=\text{cache.shape}[:-k]\). All arithmetic is float64; return values follow `cache`'s dtype. Return a dict of exactly 5 keys:

- `outlier_mask`: bool, same shape as `cache`;
- `max_error`: float, shape `lead` (the key branch broadcasts the scalar across every unit);
- `n_bits_clamped`: int64, shape `lead`, after ceil+clamp and before the trailing overwrite;
- `n_bits`: int64, shape `lead`, after the trailing overwrite, the one actually used for quantization;
- `quant_dequant`: float, same shape as `cache`, the quantize-dequantized result, with outlier positions equal to the originals.

Raise `ValueError` on: `cache` not 5-D or not float32/float64; an illegal `key_or_value`/`level`/`method`; `symmetric` not a bool; `outliers_ratio` outside \([0,1)\); `last_n_attentions < 1`; a bit-width range out of bounds or `n_bits_min > n_bits_max`; `target_quantization_error <= 0`. For the key branch: missing `q_norm`, `q_norm <= 0`, or `seq_len < 2`. For the value branch: `attentions` missing / not 5-D, its layer/batch/head disagreeing with `cache`, its last dim not equal to `seq_len`, or `q_len < last_n_attentions`. For normal: missing tables, a table list short of `n_bits_max + 1` entries, or entry \(n\) whose length is not \(2^{n}\). When a unit has too few non-outlier elements (uniform needs `>= 1`, std needs `>= 2`).

## Example

```
cache  shape (1, 3, 1, 1, 4)      # token level: one 4-element unit per (batch, seq_len)
= [[[-3.0, -0.5, 1.0, 4.0],
    [-2.0, 0.25, 0.5, 6.0],
    [-1.0, -0.25, 0.5, 3.0]]]
key_or_value="key"   level="token"   symmetric=True   outliers_ratio=0.5   method="uniform"
last_n_attentions=2  target_quantization_error=0.01   q_norm=3.0   n_bits_min=1  n_bits_max=4

max_error      = [[0.0734599093, 0.0734599093, 0.0734599093]]   # scalar, broadcast to (1, 3)
outlier_mask   = [[True, False, False, True]] * 3               # x <= 1st-smallest | x > 3rd-smallest
n_bits_clamped = [[4, 3, 3]]
n_bits         = [[4, 3, 4]]                                     # last token row forced to n_bits_max
quant_dequant  = [[[-3.0, -0.5, 0.875,  4.0],
                   [-2.0,  0.25, 0.375, 6.0],
                   [-1.0, -0.25, 0.4375, 3.0]]]
```

In each row the lowest and highest elements are flagged as outliers and kept verbatim, while the two middle ones get quantized. The last row inverts to 3 bits, but `last_n_attentions=2` unconditionally lifts the final token to 4 bits, so 0.5 tops out at the positive bound under the 4-bit step and dequantizes to 0.4375 instead of the 0.375 a 3-bit width would give.

## Constraints

- **The two thresholds compare asymmetrically**: `mask = (cache <= lower) | (cache > upper)` with `lower = kthvalue(ceil(ratio/2*M))` and `upper = kthvalue(floor((1-ratio/2)*M))`. At `ratio = 0` it short-circuits to all-False and never calls `kthvalue`.
- **Keep the two bit-width keys apart**: `n_bits_clamped = clamp(ceil(log2(Delta/(2*eps)+1)), n_bits_min, n_bits_max)`; when `last_n > 1` you then run `n_bits[:, -last_n+1:] = n_bits_max`, and only `n_bits` drives quantization.
- **The key branch's error is cache-independent**: after broadcasting every unit holds the same value (the test asserts `min == max`); the value branch reads only the last `last_n` attention rows, so changing earlier history rows does not affect the bit-width.
- **symmetric only zeroes the quantization center**: the scale becomes `2*max|x|`; but `method="normal"` still computes the standard deviation around the real mean, with ddof=1.
- **The bit-width uses ceil, not round-to-nearest**; the quantization rounding is round-half-even, then clamped to `[-(2**(n-1)), 2**(n-1)-1]`. A 1-bit symmetric quantizer can only represent `[-scale, 0]`, so every positive number collapses to 0.
- **normal buckets via searchsorted**: std-normalize, use `searchsorted(upper)` to find the bucket, then take `center[bucket]` and de-normalize; the tables must cover up to `n_bits_max`.
- **dtype and side effects**: compute in float64, cast `max_error`/`quant_dequant` back to `cache.dtype`, keep both bit-width keys int64; do not modify the input in place, and non-contiguous inputs must match a contiguous copy.
