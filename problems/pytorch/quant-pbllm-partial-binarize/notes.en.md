# Quantization: PB-LLM Partial Binarization — Algorithm Notes

## 1. The blind spot of low-bit quantization

Compressing all weights to 1–2 bits does not spread error evenly: a few large-magnitude weights (salient weights) dominate an LLM's output, and binarizing them together with many small weights causes a sharp capability drop. The observation behind PB-LLM (Partially Binarized LLMs, 2023, arXiv:2310.00034) is that small-magnitude weights are more redundant and can tolerate binarization. Spending the low-bit budget on them while keeping large weights at a higher bit width achieves the same average bit width with substantially lower perplexity.

The upstream pipeline has three steps: select a `low_frac` fraction by magnitude as the low group; represent the low group with xnor (1 bit plus mean/scale); quantize the rest to `high_bit` bits asymmetrically; combine by mask. This problem implements the RTN branch (`disable_gptq=True`) and skips Hessian error compensation.

## 2. Mask and low path

The mask threshold comes from a global sort: `thresh = sorted(|W|.flatten())[int(numel * low_frac)]`, `mask = |W| <= thresh`. Using `<=` rather than `<` lets ties enter together, so the actual fraction can be slightly above `low_frac`; with `low_frac=0` only the minimum-magnitude value is selected (possibly several elements).

The low path is calibrated on `masked = W * mask`:

\[
\mu=\operatorname{mean}(W\odot m),\qquad
s=\operatorname{mean}(|W\odot m-\mu|)
\]

Two details matter:

- Non-selected positions participate as zeros in the `mean`, whose denominator is the full element count `C` (upstream `LowQuantizer.calibrate` calls `mean(-1)` directly on `W*mask`);
- `s` is the **mean absolute deviation**, not a variance or standard deviation.

Reconstruction is `sign(w-μ)·s + μ`, a centered binary/ternary form: `sign` gives ±1 or 0, `s` sets the magnitude, and `μ` shifts the center back. Since most positions are zero after masking, `μ` and `s` are both small, matching the scale of small weights.

## 3. High path

The high path is per row (`perchannel=True`, `sym=False`):

\[
x_{\min}=\min(\min(row),0),\quad x_{\max}=\max(\max(row),0)
\]

If both are 0, they become `(-1,+1)`. `maxq = 2^bits - 1` is an all-ones unsigned code, so:

\[
s_h=\frac{x_{\max}-x_{\min}}{maxq},\qquad z_h=\operatorname{round}\!\left(\frac{-x_{\min}}{s_h}\right)
\]

Quantization is `q = clamp(round(w/s_h) + z_h, 0, maxq)` and reconstruction is `(q - z_h)·s_h`. Unlike symmetric quantization, both extremes occupy a full endpoint, and the zero-point aligns the minimum with code 0.

## 4. Implementation contract and boundaries

- **Single group**: this problem fixes `groupsize=-1`, so the threshold is taken over the entire `W`; low `mean/scale` are still per row. Smaller `groupsize` values in upstream compute a threshold per column group, which this problem does not cover.
- **RTN only**: no GPTQ cross-column compensation; the result is the pure RTN version of "mask, two-path quantization, combine".
- **All-equal weights**: the threshold equals that value, so the whole matrix enters the low path; `masked` equals the original values, `μ` equals the constant, `s=0`, and the reconstruction is exact.
- **All-zero rows**: the high path uses `(-1,+1)`, giving `s_h=2/maxq`, `z_h=round(maxq/2)`; if the row is covered by low, the reconstruction is still 0.
- **Rounding**: `high_zero` and `round(w/s_h)` use round-half-to-even (matching `torch.round`); float64 internally keeps judging stable.
- `low_frac ∈ [0,1)`: a value of 1 makes the sort index out of bounds (upstream has no margin either), so this problem raises `ValueError`.

## 5. Division of labor within the collection

- `quant-squeezellm-sensitive-kmeans`: non-uniform clustering for **sensitive channels**; this problem splits by magnitude into a binary/high-bit mix;
- `llm-int8-outlier-split`: separates outlier channels; this problem does not split channels but masks elements individually;
- `quant-awq-clip-scale-search`: scale search for salient channels; this problem uses no activation-side analysis;
- `symmetric-quantize` / `asymmetric-quantize`: uniform integer grids, the basis of the high path here;
- This problem: **magnitude mask + xnor low path + per-row asymmetric high path + mask-based combination**.

## 6. Validation

- `check_example`: hand-computed 6-element example covering the global threshold, low statistics and high reconstruction.
- `check_mask_threshold`: matches an independently computed threshold/mask for several `low_frac` values; all-equal weights select everything.
- `check_low_stats`: `low_mean/low_scale` match per-row statistics on `W*mask` (non-selected positions count as zeros).
- `check_high_stats`: `scale/zero` on asymmetric, all-positive, all-negative and all-zero rows.
- `check_matches_oracle`: 30 random configurations (including one-sign and constant rows) compared item-by-item with a pure-Python oracle.
- `check_piecewise_selection`: reconstruction equals the low/high path selected by the mask, element by element.
- `check_all_equal_and_zero_rows`: degenerate behavior of all-equal and all-zero rows.
- `check_validation`: invalid `high_bits`, `low_frac` and `W` raise `ValueError`.
