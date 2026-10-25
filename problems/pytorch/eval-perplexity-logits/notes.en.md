# In Depth: Perplexity from Logits

## Origin
- Mikolov et al., *The Recurrent Neural Language Model* (2012): PPL \(=\exp\big(\frac1N\sum_{i=1}^{N}\text{NLL}_i\big)\) where N is the number of evaluated tokens.
- HuggingFace `evaluate` and `examples/pytorch/language-modeling`: compute the mean NLL (a `CrossEntropyLoss(ignore_index=…)` reduction) and then exponentiate — exactly this problem's convention.
- Difference from `cross-entropy-loss` / `to-label-smoothing-ce`: those problems produce a **training loss tensor**; this one produces an **evaluation metric**, where the right shift, the valid-position count, the empty-set sentinel and the micro average all have to be made explicit for the first time.

## Core points and pitfalls
1. **Shift right, do not skip the shift or shift the other way**: in a causal LM `logits[b, i]` is the distribution over `token[i+1]` after seeing `token[0..i]`, so the pairs are
   \[ \big(\text{logits}[b, i],\ \text{labels}[b, i+1]\big),\quad i = 0 \dots L-2. \]
   At most \(B(L-1)\) positions exist. What gets thrown away is the **last logit position** and the **first label** — opposite ends, so a reversed shift is wrong everywhere.
2. **Gather the log-softmax, do not exponentiate then take a log**: \(-\log\mathrm{softmax}(z)_y = \log\sum_v e^{z_v} - z_y\). In code, subtract the row max \(m\) first: \(\text{nll} = \big(m + \log\sum_v e^{z_v-m}\big) - z_y\). Writing `log(softmax(...))` underflows to \(\log 0\) for sharp distributions.
3. **`ignore_index` applies to the shifted labels only**: `labels[b, 0]` never participates, whether or not it is \(-100\); prompt masking must already be baked into `labels` (see `data-collator`, `sft-assistant-mask`). Do not invent a second mask on the logits side — two masks plus one shift will double-subtract positions.
4. **\(-100\) must not be fed to `gather`**: indices have to be in \([0, V)\). Compute the mask first, read the value at `labels.clamp(min=0)`, then filter with the mask. (`F.cross_entropy(..., ignore_index=-100)` is safe because PyTorch skips those rows internally, but a hand-written `log_softmax(...)[..., labels]` silently wraps around and picks the wrong element.)
5. **The denominator is the valid count, not B·L**: padding and prompt are not counted. Averaging over all entries instead of only the valid ones treats the ignored positions as zero loss and therefore **underestimates** perplexity.
6. **micro vs macro**: this problem pools every valid token into one average. Averaging per-sample means instead (macro) gives short and long sequences equal weight and shifts the number — the same asymmetry that `eval-token-accuracy` makes its central lesson.
7. **Sentinel convention**: when \(n=0\) the metric is undefined; the choice here is \(-1.0\) rather than 0, NaN or inf, because real PPL satisfies \(\text{PPL}\ge 1\), so \(-1\) never collides with a legitimate value and stays exactly comparable (NaN interacts badly with `allclose`, inf tends to hide implementation bugs).

## Derivation (what the metric means)
\[ \text{PPL} = \exp\Big(-\frac1n\sum_{i\in\mathcal V}\log p_\theta(x_{i+1}\mid x_{1..i})\Big) = \Big(\prod_{i\in\mathcal V} p_\theta(x_{i+1}\mid x_{1..i})\Big)^{-1/n} \]
i.e. the inverse of the geometric mean of the per-position true-token probabilities. The geometric mean (rather than the arithmetic one) is precisely why exponentials and logs appear: a single catastrophic position is not diluted away.
Boundary values: fully deterministic predictions give every \(p=1\) and \(\text{PPL}=1\); a uniform vocabulary gives every \(p=1/V\) and \(\text{PPL}=V\). These two are the probes of this problem.

## Verification approach
The reference solution vectorizes: slice for the shift → `log_softmax` → `gather` → boolean index for valid positions → `exp(mean)`. The judge's `_oracle` runs three nested Python loops and computes \(\big(m+\log\sum e^{z-m}\big)-z_y\) by hand per position, accumulating sum and count (different structure, numerically equivalent), compared with `allclose(atol=1e-5, rtol=1e-5)`. Four degenerate assertions are added: all `ignore_index` → \(-1.0\); \(L=1\) → \(-1.0\); uniform logits → \(V\); one-hot confident predictions → \(\approx 1\); plus "a single valid position equals \(1/p\) at that position" and "shifting the wrong way hits the \(-1.0\) sentinel" as directed probes.

## Complexity
\((B, L-1)\) positions, each needing one \(O(V)\) log-sum-exp: \(O(BLV)\) time and \(O(BL)\) extra space (or \(O(BLV)\) if the log-softmax is materialised).

## Practical notes
Cast logits to float32 when evaluating (even if the model runs in bf16), otherwise rounding noise in `log_softmax` accumulates over long sequences; when comparing checkpoints keep the very same `labels`/mask convention, otherwise \(n\) differs and the perplexities are not comparable.
