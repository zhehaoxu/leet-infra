# Solution Notes · SmoothQuant α Search

## Sources

- SmoothQuant (Xiao et al., ICML 2023, arXiv:2211.10438) §3: the per-input-channel scale \(s_j=\max|X_j|^\alpha/\max|W_j|^{1-\alpha}\) migrates activation outliers into the weights; the paper selects \(\alpha\) by a grid search on accuracy (the implementation uses an empirical 0.5 for LLMs). This problem makes the search itself the exercise.
- The migration formula is shared with `smooth-quant-outlier-migration`, but that problem only applies one **given** α; here the question is how to choose α and what the end-to-end error after both-side quantization looks like.
- Per-token activation quantization and per-output-channel weight quantization are the standard W8A8 configuration (e.g. the int8 GEMMs in TensorRT-LLM / vLLM).

## Key points and pitfalls

**Both ends of α are traps.** At α=0, \(s_j=1/\max|W_j|\): activations barely shrink and the outlier channel still stretches the range. At α=1, \(s_j=\max|X_j|\): weights are multiplied by activation magnitudes and become harder to quantize. In the example, α=0 gives MSE 1.95e-4 and α=1 gives 1.91e-4, while α=0.5 gives 9.45e-6 — more than an order of magnitude better than either end. Trying only the two ends is not enough.

**Per token / per channel, not per tensor.** Activations use one scale per row (token), weights one per output channel. Collapsing both to per-tensor lets the outlier flatten the whole row and wipes out the α differences.

**Symmetric range is \([-q_{\max},q_{\max}]\).** Do not extend the negative side to \(-q_{\max}-1\): that shifts every rounding boundary by one code, which can change both the MSE and the winning α. Round with `torch.round` (ties to even), matching `numpy.round`.

**Short-circuit zero channels and rows.** When \(sx_j\) or \(sw_j\) is 0 the scale would be 0 or `inf`; the contract fixes \(s_j=1\). An all-zero token row must dequantize to zero, with no `0/0`.

**Divide the MSE by \(T\cdot O\).** A bare sum is not comparable across calibration shapes; the statement specifies the mean.

**Ties go to the first occurrence.** The grid may contain duplicates; update the best with a strict `<`; using `<=` silently makes ties go to the last.

**The error has two quantized factors.** \(\hat X\hat W^\top\) uses both dequantized tensors. Skipping activation quantization gives a completely different α curve — that is the diagonal W-only approximation (see `awq-activation-scale`), not this problem.

## Cross-checking

- Reference implementation: vectorized `torch`, broadcasting `sx`/`sw` per α, `_quant_sym` using `abs().amax(dim=1)` for the scale and `torch.round` for rounding, accumulated in float64.
- Independent oracle: pure-Python nested lists, per-token / per-channel `round()` (Python's banker's rounding matches `torch.round`), element-wise dot products, sharing no control flow with the reference.
- 60 random instances (\(T\le 5\), \(C\le 6\), \(O\le 4\), one channel amplified, random α subset, bits 4–8) agree bit-for-bit; plus 9 targeted cases (example, all-ones tie, activation outlier, weight outlier, zero channel, zero row, bits=4, illegal bits, scale recheck at the best α).
- Invariants: all `mses` finite, `best_alpha` in the grid, `best_scales` equal to the formula value at the best α.

## Complexity

- Per α: \(O(TC + OC + TOC)\); for a grid of size \(G\), \(O(G\cdot TOC)\) overall — milliseconds at the tested sizes.
- A real implementation batches the calibration set and searches α layer by layer; only the single-layer error accounting is kept here.

## Related problems

- `smooth-quant-outlier-migration`: the migration itself for a given α (\(s\), \(\hat X\), \(\hat W\)), i.e. this problem's inner step.
- `awq-activation-scale`: W-only α search using an activation-weighted **weight** error (diagonal approximation), with no activation quantization.
- `quant-rtn-gptq-error`: RTN vs GPTQ output MSE — another convention for measuring quantization error.
- `quant-int4-marlin-layout`: after search/quantization, how the 4-bit codes land in memory.
