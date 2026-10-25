# In Depth: Masked Token Accuracy

## Origin and engineering context
- Token-level accuracy is the directly readable argmax-hit version of `F.cross_entropy`'s per-position prediction, and it is what `accuracy` computes by default in Trax/Flax and HuggingFace `evaluate`; almost every ChatML/Llama-style SFT evaluation script prints it.
- The valid mask comes from the data-preparation problems in this bank: `data-collator` writes `ignore_index` into padding labels, `sft-assistant-mask` / `sft-turn-loss-weight` write it into the prompt segment and the multi-turn template. This problem is the **consumer** of those masks — a wrong mask still produces a plausible-looking number, so the denominator has to be counted by hand.
- Division of labour versus `eval-perplexity-logits`: that one starts from probabilities (continuous, comparable, sensitive to confidence), this one starts from argmax (discrete, right-or-wrong only). With the same model it is common for perplexity to improve while token accuracy does not move, because the probability mass merely redistributes.

## Notation and derivation
Let \(\mathcal V=\{(b,i): y_{b,i}\ne t\}\) be the valid-position set (\(t=\) `ignore_index`) and \(h_{b,i}=\mathbb 1[\hat y_{b,i}=y_{b,i}]\).
\[ \text{micro}=\frac{\sum_{(b,i)\in\mathcal V} h_{b,i}}{|\mathcal V|}, \qquad
\text{macro}=\frac{1}{|\mathcal B|}\sum_{b\in\mathcal B}\frac{\sum_{i:(b,i)\in\mathcal V} h_{b,i}}{|\mathcal V_b|}, \quad \mathcal B=\{b:|\mathcal V_b|>0\} \]
Their relation: micro is the \(|\mathcal V_b|\)-weighted mean of the per-sample accuracies,
\[ \text{micro}=\sum_{b\in\mathcal B}\frac{|\mathcal V_b|}{|\mathcal V|}\,\text{acc}_b, \qquad \sum_b \frac{|\mathcal V_b|}{|\mathcal V|}=1 . \]
So micro and macro coincide only when all \(\text{acc}_b\) are equal (or all \(|\mathcal V_b|\) are equal); in general \(|\text{micro}-\text{macro}|\) is of the order of the covariance between per-row accuracy and per-row valid count — long-and-good rows pull micro up, long-and-bad rows push it down.

## Core points and pitfalls
1. **Do not shift**: `pred_ids` is already position-aligned with `labels` (the shifting happened upstream, when the argmax over LM logits was taken). Shifting again misaligns every row and the metric collapses towards 0.
2. **The mask belongs to the labels side**: a common bug is `mask = pred_ids != -100` — predictions are vocabulary ids and never \(-100\), so the denominator silently becomes \(B\cdot L\) and padding is scored as a mistake.
3. **\(-100\) equals \(-100\), and that is a real trap**: the mask must apply to numerator **and** denominator. Writing `hit = (pred == labels)` without `& valid` donates free hits whenever `pred_ids` also holds \(-100\) at ignored positions (some pipelines copy the labels), while forgetting the denominator scores those positions as errors. Both directions are wrong.
4. **The macro denominator is the number of samples that have valid positions**: a prompt-only sample must not even be counted as a sample, otherwise its undefined \(0/0\) is treated as \(0\) and drags macro down — the same handling as "an empty group produces no advantage" in `to-grpo-advantage`.
5. **micro is not a synonym for macro**: short answers diluted by long ones is the classic misreading when a batch has few samples; state the convention when reporting, and keep it fixed when comparing experiments.

## Sentinel convention
The empty set returns \((0.0, 0.0)\), matching HuggingFace `evaluate`'s `accuracy` on empty input. The benefit is that the value always stays inside \([0,1]\) and never breaks downstream plotting; the cost is the collision with "all predictions wrong", so the test suite covers both `all_wrong` and `no_valid` — an implementation cannot tell them apart from the return value, only from the internal valid counter (the `total_valid == 0` branch).

## Verification approach
The reference solution uses tensor reductions: `valid = labels != ignore_index`, `hit = (pred == labels) & valid`, global sums divided for micro; per-row `sum(dim=1)` followed by boolean indexing with `per_valid > 0` and `.mean()` in double precision for macro. The judge's `_oracle` is a pure-Python double loop comparing element by element and collecting per-row accuracies (completely different structure, numerically equivalent); every case is compared with `allclose(atol=1e-5, rtol=1e-5)` on float64 tensors. Five further degenerate/directed assertions cover: the empty set, a perfect prediction, micro \(=\) macro for equal valid counts, the divergence for unequal valid counts (hand-computed 0.4 vs 0.625), and the requirement that whatever is written at ignored positions cannot change the result.

## Complexity
\(O(BL)\) time and \(O(B)\) extra space for the per-row counts.

## Practical notes
In dialogue data the supervised span is often a small minority of the sequence (prompt can exceed 80%), and then macro and micro can differ by several percentage points. If you want turn weighting, use something like `sft-turn-loss-weight` and compute a weighted accuracy — do not invent a new denominator on the fly.
