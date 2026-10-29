# Solution Notes · AWQ Scale-and-Clip Joint Search

## Sources

- AWQ (Lin et al., MLSys 2024, arXiv:2306.00978): per-input-channel scaling protects weights connected to large activations, and the paper also uses **clipping** before weight quantization as a second tool against outliers; both are searched together.
- The AutoAWQ implementation splits the two into the scale grid of `awq/quantize/auto_scale.py` (with the `max_shrink` bound) and the clip grid of `awq/quantize/auto_clip.py`; `max_shrink` limits how hard the scaling may squeeze the weights.
- **Scope note**: this problem's normalization (geometric mean), clip threshold (layer-global absmax) and error weight (`mean|X|` to the first power) do not match every upstream detail item by item; the formulas in the statement are the sole contract. Byte-level agreement with a specific AutoAWQ revision would require re-checking that revision's `auto_scale.py` / `auto_clip.py`.

## Key points and pitfalls

**The clip threshold is global to the layer.** \(t=\text{clip}\cdot\max_{o,j}|W_{oj}|\). With a per-row absmax, `clip=0.5` halves every row — including rows with no outlier — and the example's loss jumps from 1.4e-4 to the 1e-2 range. A global threshold only touches elements near the global maximum.

**Normalize by the geometric mean, before shrinking.** \(s/\sqrt{\max(s)\min(s)}\) keeps the overall magnitude; then apply `max(s, 1-max_shrink)`. Reversing the order (shrink, then normalize) scales the bound away and makes `max_shrink` meaningless.

**`max_shrink` is a lower bound.** \(s_j\ge 1-\text{max\_shrink}\) limits how much a weight may shrink. `max_shrink=0` means no shrinking at all, so every \(s_j\ge 1\). Do not turn it into an upper bound or read it as "shrink down to zero".

**Compute the error in the logical domain.** Quantization acts on \(W^c\odot s\), so after dequantizing you must divide by \(s_j\) to compare against \(W^c\). Skipping that lets the error drift with \(s\) and misplaces the optimal \(\beta\).

**Group boundaries and all-zero groups.** One scale per `group_size` consecutive input channels, with no borrowing across groups; an all-zero group uses scale 1 and all-zero codes, never `0/0`.

**Ties go to the first occurrence.** β outer, clip inner, update with a strict `<`; the grids may contain duplicates.

**The returned codes belong to the scaled weights.** `best_codes` are the codes of \(W^c\odot s\), not of \(W\); `best_scales` is the effective \(s\) after normalization and shrink, not the raw one.

**`xm` is a mean, not an absmax.** The importance weight is \(\frac1T\sum_i|X_{ij}|\); switching to absmax changes every loss value.

## Cross-checking

- Reference implementation: vectorized `torch`; degenerate channels handled with a boolean mask, `Wc = clamp(W, -clip*wmax, clip*wmax)`, groups via `reshape(O, C//g, g)` + `amax(keepdim)`, and a broadcast-weighted mean error.
- Independent oracle: pure-Python nested loops, per (β, clip) hand-written clamp, per-group `round()` and element-wise accumulation, sharing no control flow with the reference.
- 60 random instances (\(O\le 3\), even \(C\le 4\), \(T\le 4\), one amplified activation column, random `max_shrink`, group and bit width) agree on the five-tuple; plus 9 targeted cases (example, monotone clip improvement, β beating 0, shrink bound, group-quantization edges, bits=8, all-zero weights, illegal bits, loss table vs best pair).
- Invariants: `losses` has the right shape and the best pair attains its minimum, `best_scales` matches the `_norm_s` formula, codes lie in \([-q_{\max},q_{\max}]\).

## Complexity

- Per (β, clip): \(O(O_\text{out}\cdot C)\) quantization and error work; overall \(O(R\cdot K\cdot O_\text{out}\cdot C)\) for grids of sizes \(R\), \(K\).
- Real AWQ runs the same search with calibration forward passes; the diagonal approximation here collapses the forward into a per-channel weight, keeping the grid and clip accounting.

## Related problems

- `awq-activation-scale`: AWQ's single-parameter \(\alpha\) search with per-tensor RTN and \(x_{bar}^2\)-weighted weight error; no clipping, shrink or groups.
- `smooth-quant-outlier-migration` / `quant-smoothquant-alpha-search`: migrate the quantization burden into the weights; this problem only scales the weights and keeps activations in floating point (W4A16).
- `group-wise-quantize`: the numerical group-quantization rule itself.
- `quant-int4-marlin-layout`: how the resulting 4-bit codes land in memory tile by tile.
