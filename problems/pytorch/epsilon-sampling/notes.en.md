# In Depth: Epsilon Sampling Relative Threshold Cut

## Source

- The algorithm: Hewitt, Manning, Liang, *Truncation Sampling as Language Model Desmoothing*, arXiv:**2210.15191** (verified; Findings of EMNLP 2022), official experiment code `john-hewitt/truncation-sampling`. The paper reads truncation sampling as "desmoothing": a language model's softmax is naturally over-smoothed (the long tail comes from label-smoothing-like effects in training), and truncation's job is to **move that probability mass away** — the inspiration for this problem's `renorm=False` convention (survivors untouched, mass to the peak). Note: the threshold convention of the paper's original \(\varepsilon\) truncation and of HF's `EpsilonLogitsWarper` (absolute vs relative) is **to be re-sourced**; this problem pins the **relative** threshold \(\varepsilon\cdot p_{\max}\) — it keeps one \(\varepsilon\) comparable across distributions of different temperature/sharpness.
- Library side: HuggingFace transformers' `EpsilonLogitsWarper` (`epsilon_cutoff`); llama.cpp has no such operator (it lives in the min-p/xtc family). This problem's implementation details (equality direction, fill-the-hole tie-breaking, inverse-transform fallback) follow the sampling problems' shared conventions in this collection.
- One-line summary: **epsilon sampling = tail removal with an "absolute probability on a relative anchor"**. The threshold is neither a fixed number nor a fraction of the peak alone — it is "\(\varepsilon\) times the peak probability", combining both kinds of comparability.

## Full derivation and index convention

**Why a relative threshold**: an absolute \(\varepsilon\) barely cuts on cold (sharp) distributions and slashes huge swaths on hot (flat) ones — the semantics of \(\varepsilon\) drift with the distribution. Anchoring to \(p_{\max}\) makes the threshold **scale with the peak**: large \(p_{\max}\) (spike) → high threshold → more cuts; small \(p_{\max}\) (flat, \(p_{\max}\to1/V\)) → low threshold → fewer cuts. Two endpoints pin the behavior: \(\varepsilon=0\) keeps everything (threshold 0, \(p\ge0\) always true); \(\varepsilon=1\) keeps only the argmax (only \(p_{\max}\ge p_{\max}\)).

**Equality direction**: the test is \(p_i\ge\text{thr}\). An implementation with \(>\) cuts the \(p_{\max}\) token itself exactly at the boundary case \(\varepsilon=1\) (\(p_{\max}\ge p_{\max}\) holds with equality) → an all-zero `filtered` → division by zero in the renormalization. Equality-keeps makes "the argmax always survives" a structural guarantee and the denominator always positive.

**The essential difference between the two mass-handling modes**:

- `renorm=True` (conditional-distribution view): truncation redefines the distribution over the kept set — survivors scale up proportionally; the message is "choose among these".
- `renorm=False` (desmoothing view): truncation removes mass the model should not have emitted — the survivors' probabilities **are the model's real beliefs** and must not be artificially inflated; the cut mass has one natural destination: the distribution's optimum (argmax). This preserves both the relative order and the absolute values of all other tokens; only the peak grows. (The paper's own variant — **to be re-sourced** — may not redistribute at all, leaving a sub-stochastic distribution; this problem pins the destination to the argmax so that `filtered` is always a legal distribution.)

**Tie-breaking**: `renorm=False` dumps the hole onto `argmax`; torch's `argmax` takes the **smallest index** on ties — the judge's oracle uses the same ruling; do not invent "last index" or "split evenly".

**Inverse transform**: the collection-wide convention — CDF in original order, first prefix sum \(\ge u\) hits, fallback to the last index. Under `renorm=False` the argmax's probability is inflated, the CDF's front segment steepens, and small `u` values hit the peak more easily — the observable behavioral consequence of desmoothing.

## Key points and common mistakes

1. **The threshold anchors on \(p_{\max}\) in the probability domain** — not the logit domain (that is top-nσ), not cumulative mass (that is top-p). Mixing the three anchors is the classic cross-problem slip.
2. **Under `renorm=False` the non-argmax survivors are bit-identical**: the judge asserts equality with the raw softmax's components at `atol=1e-7`. Any implementation that "conveniently renormalizes everything" shifts every component and fails. The hole flows only to the argmax (a single-point `scatter_add`).
3. **\(\varepsilon=1\)'s one-hot**: only the argmax survives; under `renorm=False` it receives \(p_{\max}+(1-p_{\max})=1\), under `renorm=True` it gets \(1/1=1\) — the two conventions converge. With the equality direction reversed, this case divides by zero.
4. **\(\varepsilon=0\)'s identity**: threshold 0, \(p_i\ge0\) always true; the hole is 0 and peak-filling is a harmless no-op. Assert against the raw softmax directly (`atol=1e-6`).
5. **argmax is decided on the original probabilities** (`probs.argmax`), not `filtered.argmax` — filling enlarges the peak without changing the argmax, but computing argmax before filling is the safe order.
6. **Batch broadcasting**: `p_max`, threshold, argmax, and CDF are per-row; `u`'s shape follows the batch dims (scalar logits with scalar `u`).
7. **float32 details**: cast `keep` before multiplying; `scatter_add_`'s index shape is (batch, 1); the fallback index exists only for "the float prefix sum just misses \(u\)".
8. **No randomness**: `u` is the sole randomness source; any internal `torch.rand` fails the comparison.

## Differences from neighboring problems

- **vs `min-p-sampling`**: also "relative threshold + peak anchor", but different in what the anchor means — min-p's threshold \(p_{\max}\cdot\alpha\) cuts the **tail** with the threshold scaling **proportionally** to the peak; epsilon's \(\varepsilon\) is also a fraction of the peak, but its semantics are "a floor on the probability itself" rather than "a cap on the ratio to the peak". Numerically min-p keeps \(p_i\ge\alpha p_{\max}\) and epsilon keeps \(p_i\ge\varepsilon p_{\max}\) — **the criteria look identical**! The real differences are mass handling and positioning: min-p comes from llama.cpp (renorm only), epsilon from the desmoothing paper (renorm switchable, mass to the peak); min-p's \(\alpha\) is conventionally small (0.05–0.2) while epsilon's \(\varepsilon\) is chosen directly on the probability scale. The two problems' example matrices differ, and "are non-argmax survivors bit-identical?" settles it in one probe.
- **vs `top-p-sampling`**: top-p's cut point is decided by **accumulation** (no matter how heavy the head, stop at p), epsilon's by **single-point probability** (cut the whole tail, keep the whole head). On spiked distributions top-p collapses to 1–2 tokens early, while epsilon keeps the full batch of high-probability tokens — opposite attitudes toward the head.
- **vs eta-sampling (the paper's entropy-based sibling)**: eta adapts the threshold to the entropy, \(\min(\varepsilon,\sqrt{\varepsilon}\,e^{-H})\); this problem's threshold has no entropy term — purely statistical.
- **vs `dripping-sampling`**: statistical truncation in the opposite direction — dripping cuts the **head** (above the mean / mean+σ), epsilon cuts the **tail** (below a fraction of the peak). One problem's survivor set is exactly what the other compresses.

## Cross-check convention and complexity

- Main criterion: the independent recompute `_oracle` — pure-Python loops: term-by-term softmax, hand-computed threshold, explicit \(\ge\) filtering, both mass-handling modes (renorm by division, fill-the-hole by a single-point add), and a while-loop prefix-sum inverse transform. Compared against the reference (torch-vectorized + scatter_add) over 40 random batch-logit trials × {True,False} × {0.05,0.2,0.5,0.9} full grids: filtered `allclose(1e-5)`, chosen `torch.equal`.
- Dedicated assertions: the two endpoints \(\varepsilon\in\{0,1\}\) (identity / one-hot); under `renorm=False` "non-argmax components untouched" (`atol=1e-7`); mass conservation `filtered.sum()==1` under `renorm=False`.
- Complexity: \(O(V)\) per row (softmax, threshold, filtering, renormalize/peak-fill, CDF — one pass each), no sorting; memory on the order of the input.
