# Solution Notes · QAQ Error-Aware KV Bit-Width Solving

## In one sentence

Set a per-unit error budget \(\varepsilon_{\max}\), **solve for** the bit-width via \(n=\lceil\log_2(\Delta/(2\varepsilon_{\max})+1)\rceil\), then run one affine quantize-dequantize at that width; outliers are removed first and kept verbatim.

## Why this inversion gives the width

The unit's non-outlier dynamic range is \(\Delta\), and an \(n\)-bit affine quantizer uses a step of \(\Delta/2^{n}\); round-to-nearest puts the error at about half a step. So \(\Delta/(2\varepsilon_{\max})\) reads as "how many half-error cells fit across the range", and \(\log_2\) of that, rounded up, is the number of binary cells needed to hold that many. The `+1` is what makes a constant unit (\(\Delta = 0\)) land on \(\log_2 1 = 0\), after which `n_bits_min` holds it up.

The two error budgets are QAQ's own, pinned here verbatim:

\[
\varepsilon_{\max}^{\text{key}}=\sqrt{\frac{12}{\lVert q\rVert_2}\,\log\Bigl(\frac{L^{3}}{L-1}\,\tau^{2}+1\Bigr)},\qquad
\varepsilon_{\max}^{\text{value}}=\frac{\sqrt{12/L}\ \tau}{\max|A|}
\]

The only difference is **what they depend on**: the key branch eats just three scalars \((L,\tau,\lVert q\rVert_2)\) and is completely independent of the stored cache values, so one \(\varepsilon_{\max}\) serves the whole tensor; the value branch divides by the amax over the quantize dims of the last `last_n` attention rows, so it varies unit by unit — a larger amax divides down to a tighter \(\varepsilon_{\max}\), and the bit-width rises accordingly.

## Three conventions that are easy to get wrong

1. **The two thresholds compare asymmetrically.** `lower` is the \(\lceil \text{ratio}/2\cdot M\rceil\)-th smallest and `upper` the \(\lfloor(1-\text{ratio}/2)M\rfloor\)-th, then `mask = (x <= lower) | (x > upper)` — inclusive low end, exclusive high end. Writing `<=`/`<=` or `>`/`>` flips the mask exactly on the boundary values. At `ratio == 0` it short-circuits to all-False and never touches `kthvalue` (otherwise `k = M` would be out of range or select the maximum).
2. **There are two bit-width keys.** `n_bits_clamped` is the ceil+clamp result; `n_bits` additionally overwrites the last `last_n-1` tokens with `n_bits_max` (those tokens do not have `last_n` rows of attention history, so their error cannot be estimated reliably). Quantization uses the latter; the former is only an intermediate and must be returned alongside it.
3. **Outliers must be removed in two places.** The range \(\Delta\) (i.e. `scale_value`) is computed over non-outlier elements only — otherwise an extreme blows up the range and every element gets more bits; and on write-back the outlier positions pass through `torch.where(mask, x, ...)` untouched by any grid mapping.

## What is judged

- `max_error.shape == n_bits.shape` (the width shape `lead`); the key branch asserts `min() == max()`.
- `n_bits_clamped` and `n_bits` match separately, and differ only on the last `last_n-1` tokens.
- uniform: after normalizing, `round` (**half-even**, not half-up) then clamp to `[-(2**(n-1)), 2**(n-1)-1]`, finally `q*scale + mean`.
- normal: std-normalize (ddof=1) → `searchsorted(upper_table)` for the bucket → take `center_table[bucket]` → de-normalize. `symmetric` only zeroes the quantization `mean`; **the standard deviation is still around the real mean**.
- A 1-bit symmetric quantizer's grid points are `[-1, 0]`, so every positive collapses to 0 — a direct consequence of the clamp upper bound `2**(n-1)-1 == 0`, not a bug; meanwhile `quant_dequant[mask] == cache[mask]` and `n_bits.min()/max()` stay inside the given range.

## Boundaries

- The key branch with `seq_len < 2` would divide by `L-1`, so it raises; a missing or non-positive `q_norm` also raises.
- The value branch needs 5-D `attentions` of shape `(n_layer, n_batch, n_head, q_len, seq_len)` whose layer/batch/head match the cache, whose last dim equals `seq_len`, and with `q_len >= last_n_attentions`.
- Every unit needs at least 1 non-outlier element (otherwise \(\Delta\) is undefined), and at least 2 when `method="normal"` takes the std (ddof=1); the quantization loop visits only the widths actually present in `n_bits`, not the whole `n_bits_min..n_bits_max` span.
- Accumulate in float64 throughout and cast back to `cache.dtype`; `n_bits` should be identical between a float32 and a float64 input, with only the float values differing within tolerance.

## Complexity

With \(U\) quantization units and \(M\) elements per unit: every reduction (`kthvalue`, amax/amin, mean/std) is linear within a unit, totalling \(O(UM)\); quantization groups by bit-width and runs one vectorized normalize-round-denormalize per group, with normal paying an extra searchsorted cost of \(O(M\log 2^{n})\). Overall this stays \(O(UM\log M)\), with no term multiplied by the maximum bit-width.

## Source

`src/quantizer.py::Quantizer` in `ClubieDong/QAQ-KVCacheQuantization` (arXiv 2403.04643):

- `__init__` L59-64: the three `quantize_dims` settings, token=(-3,-2,-1) / layer=(-2,-1) / head=(-1,);
- `_calc_quantization_bits` L148-195: the key scalar formula at L160; the value branch L163-169 keeps the last `last_n` rows, permutes to (n_batch, seq_len, n_layer, n_head, last_n) then takes amax over the quantize dims, with \(\varepsilon_{\max}\) at L171; `scale_value` symmetric L177-179, asymmetric L180-183; L187 `log2(scale/(2*max_error)+1)`; L191 `ceil -> int64 -> clamp`; L194-195 `if last_n>1: n_bits[:, -last_n+1:] = n_bits_max`;
- `_calc_outlier_mask` L198-212: L199-200 the `ratio == 0` short-circuit, L204-205 the two `kthvalue` calls, L210 `(cache <= lower) | (cache > upper)`;
- `_normalize` L215-241 / `_uniform_quantize` L246-253 / `_normal_quantize` L255-262: masked center and scale (symmetric zeroes `mean_value` at L219; std goes through masked `cache.std()` at L224/L232, i.e. ddof=1); uniform is minmax scale + `round` + clamp; normal is std scale + `searchsorted` into the upper table then indexing the center table; both let outliers through via `torch.where(outlier_mask, cache, ...)`;
- `quantize` L269-284: L269 `cache.permute(1, 3, 0, 2, 4)` into this problem's input layout, L279-281 the per-width-group write-back `cache[indices] = method(...)`.

**Additions of this problem** (not in upstream, needed by the judging environment — do not credit them to the paper or the repo): **(a)** `max_error` is broadcast to the same per-unit shape as `n_bits`; upstream's key branch is a bare Python scalar (broadcast implicitly by the division), and here it is explicitly laid out as a tensor with the per-unit equality asserted; **(b)** the `method="normal"` quantile tables were originally produced by `scipy.stats.norm.ppf` in `set_dtype_and_device` L117-124, but scipy is unavailable in the judging environment, so they become explicit arguments `quantile_upper`/`quantile_center`; **(c)** `quantize`'s returned `average_n_bits` and the cache-size accounting in `calc_quantized_cache_size_per_token` L286-298 are out of scope; **(d)** upstream leans on `assert` (the L51-95 parameter checks), which this problem uniformly turns into `ValueError`. Also, the inputs here are already in the post-permute layout and `attentions` is handed in as a stacked 5-D tensor, with no model run.

## Related problems

- `kv-quest-page-bound`: also "extremes in place of real computation", but Quest uses them for page selection while this problem uses them to set each unit's storage width; `quant-kivi-kv-asym` is likewise asymmetric KV quantization, but with a fixed width, whereas here the width is solved per unit.
- `quant-gear-lowrank-outlier`: GEAR compensates outlier quantization error with a low-rank term; this problem simply skips outliers and keeps their originals, with no compensation at all.
