# In Depth: Dripping Ultra-High Probability Cut

## Source

- The algorithm: **to be re-sourced**. The "cut the ultra-high-probability head" family of behaviors appears in experimental samplers across inference engines/community tooling (the anti-spike idea dual to min-p); no verifiable original paper or authoritative implementation name was found, and the offline judge has no access either. So **the algorithm is defined entirely by the problem statement and starter comments**: valid-entry statistics, population std, two threshold modes, strict-greater cut, and the "must cut and must have survivors" firing rule.
- Half of the mechanism's lineage is `top-n-sigma` (already in this collection): that problem uses a \(\mu\pm n\sigma\) threshold in the **logit domain** to cut the tail; this problem moves the statistics into the **probability domain**, reverses the direction to cut the head, and adds the firing backstop. The other half is the shared `min_keep` backstop semantics of llama.cpp's truncation operators.
- One-line summary: **min-p asks "how much shorter than the peak before we cut", Dripping asks "how much taller than the average"**. The former's reference point is the maximum (tail truncation); the latter's is the row distribution's central tendency (head truncation) — two directions, two reference frames, a natural dual pair.

## Full derivation and index convention

**Why statistics only over valid entries**: softmax over long vocabularies produces large numbers of underflowing entries (\(p\sim10^{-40}\)). Including them in the mean/variance only adds tiny numbers, but it drags \(\bar p\) down and dilutes \(\sum p_i^2/n\)'s effective mass — the statistics are meant to describe the *competitive* candidates. This problem fixes the valid mask at \(p_i>10^{-35}\): those entries would be cut by any threshold anyway and do not deserve a say. Note the direction: the valid mask shapes the **statistics only**; the `cut` test still applies to **all** \(p_i\) (underflow entries are forever below the threshold and always survive).

**Population vs sample std**: \(\sigma=\sqrt{\frac1n\sum p_i^2-\bar p^2}\), denominator \(n\). Writing the sample std (\(\frac1{n-1}\)) systematically inflates the threshold and cuts less — a per-row difference of order 1/n, exactly resolvable by the judge's `atol=1e-5`, and aligned with `top-n-sigma`'s "population std" convention.

**Geometry of the two modes**: the `mean` mode's threshold scales with the whole row (\(\alpha\bar p\) is multiplicative), fitting "rows that are uniformly flat" — on a flat distribution \(\bar p=1/n\), every \(p_i\) equals the threshold and nobody is cut (protected by strict greater-than). The `std` mode's threshold is center \(+\alpha\sigma\), where σ measures spike prominence: the sharper the distribution, the larger σ, the higher the threshold, the harder the cut — **adaptively shaving the spike**, which is the "dripping" intuition. Both modes agree on the `drip_amount` monotonicity: larger \(\alpha\), higher threshold, smaller cut set (the judge has a dedicated monotonicity assertion).

**Two firing conditions, both mandatory**: ① `cut.sum()>0` — on flat distributions "nothing to cut" is the norm and must not count as firing; ② survivors \(\ge\)`min_keep` — with small \(\alpha\) in `mean` mode on a spiked distribution, the head cut could swallow a huge batch; the backstop stops it. When not firing, `filtered` is **bit-for-bit** the raw softmax (not "cut then refill"), and `cut_count=0`.

**Inverse transform**: the same convention as every sampling problem in this collection — `filtered`'s CDF in original order, first prefix sum \(\ge u\) hits, fallback to the last index. `u` is an explicit argument (scalar or batch-shaped), so judging is fully deterministic.

## Key points and common mistakes

1. **Cut the head, not the tail.** Typing `cut = probs < thr` turns this into a min-p/top-nσ-direction problem — the judge's `check_head_not_tail` asserts "spike at index 0 zeroed, contiguous tail alive" precisely against it.
2. **Strictly greater**: a token with \(p_i=\text{thr}\) survives. On a flat distribution every \(p_i=\bar p\); writing \(\ge\) cuts the entire row to zero survivors, which the `min_keep` backstop then "rescues" — the visible outcome matches, but `cut_count` and the intermediate state differ, and correctness would rest on the backstop rather than the criterion.
3. **Population std, valid-entry statistics**: either statistical convention (denominator \(n\), valid mask) skewed shifts the `std` threshold and fails the random comparison.
4. **Not firing ≠ empty cut**: rows with `cut_count=0` must have `filtered` bit-identical to the raw softmax — "cut 0 but renormalize" is numerically harmless (sum still 1), but "backstop-blocked fake cuts" renormalized anyway differ from the raw softmax by \(10^{-8}\)-ish and can graze `atol=1e-5`. The correct form passes the raw probabilities through untouched on unfired rows.
5. **`min_keep` means "cut only if enough remain"**: compute the cut, then check survivors — never "cut down to min_keep" (that is top-k's semantics).
6. **`mode` is a string and must be validated**: any other value raises, not silently falls back to `mean`.
7. **Batch broadcasting**: statistics, threshold, cut, firing, and the inverse transform are all per-row; `u` and `cut_count` shapes follow the batch dims (scalar logits → scalar cut_count).
8. **float32 details**: the \(10^{-35}\) valid bound, and `clamp_min(0)` so \(\sum p_i^2/n-\bar p^2\) cannot go negative (numeric error) and produce a NaN sqrt — the flat test case sits exactly on the zero-variance boundary.

## Differences from neighboring problems

- **vs `min-p-sampling`**: the head–tail dual. min-p's threshold is \(p_{\max}\cdot\alpha\) (**multiplicative, referenced to the peak**, cuts the tail); Dripping's is \(\bar p\) or \(\bar p+\alpha\sigma\) (**additive/multiplicative, referenced to the central tendency**, cuts the head). On the same distribution the two cut sets are disjoint (one in the low-probability right side, one at the peak itself); on a flat distribution min-p degenerates to "keep all", and so does Dripping (nobody above the mean) — the reversed-direction dual remains consistent even at the degenerate point.
- **vs `top-n-sigma-sampling`**: both consume \(\mu,\sigma\), with three differences — domain (logit vs probability), direction (tail vs head), backstop (no min_keep vs min_keep). Read together they cover the 2×2 quadrants of "statistical thresholds".
- **vs `top-k-sampling`**: top-k cuts the head **by rank** with k independent of the distribution's shape; Dripping cuts the head **by statistical threshold**, adaptively (spikes get cut hard, flat rows not at all). Same "cut the head" behavior, one judged by order, the other by statistics.
- **vs `adaptive-p-sampling` / `dynatemp-sampling`**: those adapt parameters across **generation feedback** (cross-step adaptivity); here \(\alpha\) is constant and the statistics come from the **current row** — within-step adaptivity, no memory across steps.

## Cross-check convention and complexity

- Main criterion: the independent recompute `_oracle` — pure-Python loops: term-by-term softmax exp, hand-counted valid entries, mean/population variance by the defining formulas, both threshold modes, explicit firing test, survivor renormalization by dividing by `sum`, inverse transform by a while-loop prefix sum. Compared against the reference (torch-vectorized) over 25 random batch-logit trials × {mean,std} × {0.2,0.5,1.0,3.0} × {1,5,9} full grids: filtered `allclose(1e-5)`, chosen/cut_count `torch.equal`.
- Dedicated assertions: flat distribution does not fire (the degenerate anchor); head zeroed, tail kept (the direction anchor); the `min_keep` backstop (`cut_count=0` and `filtered` equals the raw softmax); `drip_amount` monotonicity.
- Complexity: \(O(V)\) per row (softmax, statistics, cut, renormalization, CDF — one pass each), no sorting — cheaper than top-k/top-p; returns the full `filtered` (batch, V) snapshot.
