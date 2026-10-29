# Deep Dive: The Sink Term of SoftMax1

## Where it comes from

- StreamingLLM: Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024, arXiv **2309.17453**. Section 3.3 introduces **SoftMax1**:

  \[
  \text{SoftMax1}(x)_i=\frac{e^{x_i}}{1+\sum_j e^{x_j}}
  \]

  The paper describes this extra term as equivalent to inserting a token whose logit is constantly 0 into the keys, and reports it as a baseline for mitigating the attention sink (it helps but does not remove the dependence on the first token, which is why the final method keeps the sink token's KV explicitly).
- Boundaries with neighbouring problems in this library: `kv-streaming-llm-sink` is kvpress **cache-eviction scoring** (a binary keep table) and `nu-online-softmax-rescale` is the **(max, sum) rescaling merge of blockwise softmax**; this problem only covers the normalizer itself — one extra denominator term and row sums below one. No formula overlap.

## Core points and pitfalls

1. **The sink must join the maximum**. Per row use \(r=\max(\max_{\text{kept}} x, \texttt{sink\_logit})\). Leaving it out breaks two ways: a fully masked row makes `max` return \(-\infty\) and every later step collapses; a row far below the sink (e.g. `[-1000, -999]` with sink 0) makes \(e_{\text{sink}}=e^{999}\) overflow to inf and the weights become NaN.
2. **Dropped positions enter neither the maximum nor the denominator**, and their output must be exactly `0.0`, not "a tiny number". Replacing dropped entries with the row reference before `exp` and zeroing them with `where` avoids both `exp(large positive)` and any residue.
3. **The deficit is the sink weight**. `weights.sum(-1) + sink_weight = 1` is a hard constraint; renormalizing `weights` by its own row sum ("restoring" 1) is the canonical wrong answer, because it silently falls back to plain softmax.
4. **Fully masked rows**: all `weights` are 0 and `sink_weight` is 1, never NaN. `torch.softmax` returns NaN there; this problem deliberately lets the sink absorb everything, mirroring the engineering convention for \(\pm\infty\) rows in `nu-online-softmax-rescale`.
5. **Float64 intermediates**. A "large offset plus tiny gap" pair such as `(1e5, 1e5 - 1e-3)` collapses to `(1e5, 1e5)` in float32, degenerating the weights to 0.5; only float64 preserves the three significant digits of the gap.
6. **`sink_logit` is a global rescale**: the ratio of any two visible weights is independent of it and depends only on score differences; a larger sink shrinks every visible weight while monotonically growing the sink weight. `sink_logit = 0` is the paper's zero token, and a large negative value degenerates to ordinary masked softmax.
7. **Mask broadcasting**: an `(S,)` mask must apply to `(B, H, S)` exactly like an explicit `expand`; masking stays row-wise, so different batch rows may keep different sets of positions.

## How it is judged

- `_oracle` is a row-by-row scalar numpy recomputation: it scans the kept entries for the row maximum, then evaluates `math.exp` term by term, never calling `exp` on dropped positions — no code is shared with the vectorized solution.
- An independent equivalence test appends `sink_logit` as one extra logit column and calls `torch.softmax` (float64): the first S columns must equal `weights` and the last column must equal `sink_weight`, which is literally the paper's "one more zero token". `torch.softmax` returns NaN for fully masked rows, so that check requires one kept entry per row and the NaN convention is asserted separately by `check_all_masked_rows`.
- Structural assertions: row-sum deficit, ratio invariance in the sink, monotonicity in `sink_logit`, mask-broadcast equivalence, extreme sink magnitudes, and a float64 precision probe; plus 17 invalid inputs (scores non-tensor/integer/bool/0-dim/empty last dim/NaN/±inf, mask non-tensor/float/not broadcastable, sink_logit bool/str/None/NaN/±inf) that must raise ValueError.
- All 11 `check_*` functions pass; mutation testing on the reference solution (sink excluded from the max, sink missing from the denominator, dropped entries becoming 1, float32 intermediates, mask ignored, constant sink weight) is caught by at least one check.

## Complexity

- Time \(O(\text{number of elements})\) and space \(O(\text{number of elements})\); one maximum scan, one elementwise `exp` and one sum, with no \(O(S^2)\) structure.
