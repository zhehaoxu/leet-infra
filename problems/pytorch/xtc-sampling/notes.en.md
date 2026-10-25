# Deep dive: XTC Mid-Probability Elimination (Exclude Top Choices)

## Origin

- **There is no paper.** XTC is not an algorithm proposed by an academic work; it is an implementation technique
  inside inference engines that "points the truncation knife the other way round". This problem's specification is
  aligned directly with **ggerganov/llama.cpp**'s `src/llama-sampler.cpp`: the sampler struct `llama_sampler_xtc`
  holds the four fields `probability / threshold / min_keep / seed`, the logic lives in `llama_sample_xtc_apply`,
  it is registered by `llama_sampler_init_xtc(p, t, min_keep, seed)`, and its parameters are assembled in
  `common/sampling.cpp` from `xtc_probability / xtc_threshold / min_keep` (that file pushes
  `llama_sampler_init_xtc` into the sampler chain). The function names, field names and order of checks above were
  **verified line by line against the master-branch source in this batch**, not relayed from memory. The upstream
  semantics: ① `return` immediately if any of `probability <= 0.0f || threshold > 0.5f || cur_p->size < 2` holds;
  ② draw a `uniform_real(0,1)` `chance` and `return` if `chance > probability`; ③ softmax the candidate array and
  arrange it descending, then `for i: if p[i] >= threshold → pos_last = i else break` (`pos_last` initialized to 0);
  ④ `if (size - pos_last >= min_keep && pos_last > 0) { data += pos_last; size -= pos_last; }`.
- The other frequently cited implementation is **oobabooga/text-generation-webui**'s logit filter (`xtc_filter`,
  also with `xtc_threshold` / `xtc_min_tokens`). **Its current file path and its exact predicates could not be
  fetched from this machine in this batch — the `raw.githubusercontent.com` requests to that repository timed out
  repeatedly — so it is marked 待回源 (to be traced back)**; this problem cites none of its details.
- **Scope difference versus the title and versus the collection entry (must be stated plainly).** The collection
  entry describes XTC as "compute \(\mu,\sigma\) only over tokens with \(p>10^{-35}\), then eliminate the tokens
  whose probability falls inside the **closed band** built from those \(\mu/\sigma\) statistics". The llama.cpp
  implementation verified above **uses no \(\mu/\sigma\) statistic anywhere** and has no \(10^{-35}\) probability
  floor (it compares against one fixed absolute probability threshold \(t\)). That "remove the middle μ/σ band"
  formulation has no source in arXiv nor in the upstream code we could reach, so it is **待回源** and unrelated to
  this implementation — it is closer to the **inverse** use of the statistical truncation family that
  `top-n-sigma-sampling` covers. "Mid-probability elimination" in the title should therefore be read as: after the
  over-threshold **head prefix** is eliminated (keeping the last member of that prefix), the probability mass is
  re-occupied by the **middle and tail** candidates; what is actually deleted is the **high-probability head**,
  which is exactly the literal meaning of the upstream name *Exclude Top Choices*. The grading oracle is
  `starter.py` / `solution.py`, and they are the sole authority.
- Nothing to borrow in the grading environment either: `transformers` 5.20.2 in `.venv-eval` has no XTC-related
  `LogitsProcessor`/sampler class, llama.cpp is absent, so the prefix counting and the scatter mask must be hand-made.

## Full derivation and index conventions

One row of input \(z\in\mathbb{R}^{V}\), \(p=\operatorname{softmax}(z)\), parameters \(\pi\) (trigger probability),
\(t\) (threshold), \(m\) (`min_keep`), and explicit random numbers \(u_{\text{gate}},u\in[0,1)\). **Keep "rank" and
"index" strictly apart**: a rank is a position in the descending array, an index is a position in the original
vocabulary.

1. **Early exit** is a global switch: \(\pi\le 0\ \vee\ t>0.5\ \vee\ V<2\) ⇒ `keep ≡ 1` and `applied = False` for
   that row. The \(t>0.5\) clause is not arbitrary: probabilities sum to 1, so two candidates both \(\ge t>0.5\) is
   impossible, hence the passing set has at most one element and the removal set is necessarily empty — upstream
   simply switches the whole sampler off. With \(V<2\) the rule "keep the last one of the prefix, delete the rest"
   does not even exist.
2. **Descending ranks.** Sorting by the key \((-\,p_i,\ i)\) ascending yields `order`; the original index at rank
   \(r\) is \(\text{order}_r\). Ties must be pinned by ascending index, otherwise the `keep` mask is undetermined on
   equal-probability rows.
3. **The passing prefix.** With \(a_r=p_{\text{order}_r}\) we have \(a_0\ge a_1\ge\dots\ge a_{V-1}\), therefore
   \[
   \{r: a_r\ge t\}\ \text{is a contiguous prefix starting at }0,\qquad
   cnt=\#\{r: a_r\ge t\},\quad \text{pos\_last}=cnt-1 .
   \]
   Contiguity follows from monotonicity (once some \(a_r<t\), every later \(a\) is even smaller); that is precisely
   why "count the prefix length" is equivalent to upstream's "break at the first \(a_r<t\)". Conversely, writing the
   predicate in logit space (as `top-n-sigma` does) is still monotone but means something entirely different —
   the \(t\) of this problem is a **probability**. Edge values: \(cnt=0\Rightarrow\text{pos\_last}=-1\)
   (even \(a_0\) fails); \(cnt=1\Rightarrow\text{pos\_last}=0\).
4. **Trigger and removal.** The removal set is the rank interval \([0,\text{pos\_last})\) (**left-closed,
   right-open** — rank \(\text{pos\_last}\) itself survives), so exactly \(\max(\text{pos\_last},0)\) ranks are
   deleted and the survivor count is \(V-\text{pos\_last}\). The cut fires only when all three conditions
   \[
   u_{\text{gate}}\le\pi,\qquad \text{pos\_last}>0,\qquad V-\text{pos\_last}\ge m
   \]
   hold, and then `applied` is `True`. Note the \(\text{pos\_last}>0\) clause: it excludes "exactly one head
   candidate passes", so XTC **never leaves the argmax standing alone** (in the sample,
   \(z=[9,-1,-2,-3]\) with \(t=0.5\) is stopped right here). Implementation-wise
   `keep = 1 − [(rank < pos_last) ∧ fired]`, with the mask scattered back to original-vocabulary order via `order`.
5. **Renormalization.**
   \[
   \text{filtered}_i=\frac{p_i\cdot\text{keep}_i}{\sum_j p_j\,\text{keep}_j},
   \]
   where the denominator is the **sum of surviving probabilities** (\(0.216040\) in the sample) and the original
   vocabulary order is preserved. Because firing implies \(\text{pos\_last}\le V-1\), at least one candidate
   survives and the denominator is strictly positive.
6. **Inverse-transform sampling.**
   \[
   \text{chosen}=\min\{i:\ \text{CDF}_i\ge u\},\quad \text{CDF}_i=\sum_{j\le i}\text{filtered}_j ,
   \]
   and the CDF is taken in **original order** (not descending) — consistent with the other samplers of this
   collection. If \(\forall i,\ \text{CDF}_i<u\) (the floating-point corner where \(u\) hugs 1) the last index is
   returned. \(u=0\) always returns \(0\), even when \(\text{filtered}_0=0\): this is the problem's fixed
   convention, and measured on the sample row \(u=0\) gives `chosen = 0` (an already-removed token).
7. **Numeric self-check** (sample row, everything produced by the reference solution / the environment).
   \(p=[0.487983,0.295977,0.179519,0.024295,0.008938,0.003288]\) whose descending CDF is
   \([0.487983,0.783960,0.963479,0.987774,0.996712,1.000000]\); at \(t=0.05\) the passing count is 3,
   \(\text{pos\_last}=2\), ranks 0/1 are removed, the surviving mass is \(0.216040\), and the surviving part of
   `filtered` is \([0.830953,0.112457,0.041371,0.015219]\). Incidentally this shows that a top-p style cut
   (nucleus \(=0.95\)) keeps the rank set \(\{0,1,2\}\) whereas this problem removes \(\{0,1\}\) — XTC and nucleus
   sampling use **the same prefix notion, in opposite directions**.

## Core ideas and pitfalls

1. **The direction** is the real point of this problem. Almost every truncation sampler cuts the **tail**; XTC cuts
   the **head**. `check_cuts_head_not_tail` exists exactly to catch implementations that turned the removal into a
   retention — it asserts `filtered[0] == 0` while the tail entries `filtered[3]` and `filtered[5]` stay positive.
   Writing `rank <= pos_last` (deleting the boundary candidate too), or applying `rank < pos_last` to a **keep**
   mask, both break on this assertion.
2. **The half-open interval / keeping the boundary member.** Upstream shifts the pointer with `data += pos_last`,
   i.e. rank \(\text{pos\_last}\) **stays in** the candidate set. So the survivor count is \(V-\text{pos\_last}\),
   not \(V-\text{pos\_last}-1\); and `min_keep` is checked against the survivor count **after** the cut, not the
   candidate count before it.
3. **`pos_last > 0` versus the initial value of `cnt`.** Upstream starts `pos_last` at 0 (if the very first element
   fails the threshold it stays 0 and no cut happens); this problem starts from \(cnt-1\) (that same case yields
   \(-1\), also no cut). The two writings agree in outcome, but "exactly one candidate passes"
   (\(\text{pos\_last}=0\)) must **not** be reported as "removed 0 candidates, but applied=True" — `applied` has to
   be `False`.
4. **Do not conflate \(\pi\) with \(t\).** \(t\) decides how deep the cut goes, \(\pi\) decides whether this step
   cuts at all. The two explicit random numbers each serve their own purpose: \(u_{\text{gate}}\) only passes the
   gate and must never be reused as the sampling \(u\) (that would correlate the same row with itself and destroy
   the "usually does not fire" semantics). \(\pi=0\) means **completely off** (early exit); \(\pi=1\) means
   **always past the gate** (only the \(pos\_last\) and `min_keep` clauses remain).
5. **When the gate is closed / early exit, `filtered` must still equal the softmax.** Because keep is all ones and
   the denominator is \(\sum p=1\). Returning un-normalized logits on early exit is a common mistake; `chosen` must
   **still be produced** on early exit (with the same CDF inverse transform).
6. **Ties must be stable.** The row with \(V=2\), \(t=0.5\), \(p=[0.5,0.5]\) has both candidates passing ⇒
   \(\text{pos\_last}=1\) ⇒ remove rank 0 ⇒ `filtered=[0,1]`, `chosen=1`: **the argmax is cut away and the step
   becomes a deterministic pick**. A different tie-break from the reference solution yields a different removal
   interval on such exactly-tied rows.
7. **Do not mix logits and probabilities.** \(t\) is compared against \(p\) (after the softmax). Writing the
   predicate as `logit >= t` or `log_softmax >= t` gives a completely different removal set under high or low
   temperature.
8. **Batching and reshape.** The shapes of `applied` / `chosen` are `(...,)`; after flattening `u_gate` you must
   `reshape(shape)` it back onto the row dimension. In `(..., V)` the \(V\) is only the last dimension, the
   \(V<2\) early exit tests that dimension, and it has nothing to do with the batch dimensions.

## Differences from neighbouring problems

- **vs `top-n-sigma-sampling`** (the most easily confused pair, and the core contrast of this problem). Same
  "adaptive truncation" family, but **both the criterion and the direction are inverted**: `top-n-sigma` works in
  **logit space**, builds an upper band \([\max-n\sigma,\ +\infty)\) from the row's \(\mu,\sigma\), **keeps** the
  statistically upper candidates and clears the long tail (the argmax is always inside the kept set); this problem
  uses **no \(\mu/\sigma\) at all**, works in **probability space** with a fixed threshold \(t\), finds the few
  most-head candidates above it and **removes** them (keeping that boundary one), while the entire tail survives.
  In one line: `top-n-sigma` **keeps** the upper band, XTC **cuts** the upper band. Also note: the collection entry
  phrased XTC as "remove the middle of a μ/σ closed band", which this batch showed in the Origin section does not
  correspond to any implementation we could verify.
- **vs `min-p-sampling` / `top-p-sampling`**: both of those **cut the tail** using the top probability as reference —
  `min-p` keeps everything above the floor \(p\cdot p_{\max}\), `top-p` keeps the prefix up to cumulative mass \(p\)
  and deletes everything after it; they share the property that **the head always survives** (in `min-p` even the
  argmax itself is in the kept set). This problem is the mirror image: the head is the **only** thing being cut and
  no tail element is deleted. On the same sample row, `top-p(0.95)` keeps ranks \(\{0,1,2\}\) while this problem
  removes \(\{0,1\}\) and keeps \(\{2,3,4,5\}\).
- **vs `top-k-sampling`**: there, \(k\) means "how many to keep" (a rank-based notion), the kept set is renormalized
  and then a lottery is drawn; here \(t\) means "how high is absurdly high" (an absolute-probability notion), and the
  **trigger itself is random** (the \(\pi\) gate), with the whole cut abandoned when `min_keep` is not met —
  `top-k` has neither of those two switches.
- **vs `typical-sampling`**: typicality sampling grows a window outward from the **centre** by
  \(|-\log p-H|\) and keeps candidates that are "neither too surprising nor too typical"; it can drop the argmax,
  but what it drops are the **least surprising** entries. XTC drops the **most probable** entries and never looks at
  the entropy.
- **vs `penalty-sampling` / `dry-sampling` / `contrastive-sampling`**: that family suppresses repetition using the
  **history** (n-gram counts / a Z-algorithm / embedding similarity), whereas XTC is **entirely history-free** and
  only sees the probability shape of the current step; `contrastive-sampling` does not delete any candidate at all.
  Both fight repetition, but XTC does it by "don't let the most certain few win every time".
- **vs the combined pipeline (e.g. `llama-sampling-pipeline`)**: upstream's default chain places xtc **after**
  `top-k / top-p / top-n-sigma / min-p` and **before** `typical-p / temperature` (the assembly switch in
  `common/sampling.cpp` lists exactly this enum order); and inside `llama_sample_xtc_apply` the code performs its
  **own** `llama_sampler_softmax_impl(cur_p, true)` (compute probabilities, keep the array descending) before
  scanning the prefix, does **not** renormalize afterwards (that is left to the following samplers in the chain),
  and does **not** pick a token. This problem merges those three concerns into one step: an explicit softmax, the
  surviving set renormalized and returned as `filtered` in original vocabulary order, and an explicit \(u\) driving
  `chosen` directly — with a separate assertion for each of \(V\), \(t\), \(\pi\) and `min_keep`.

## Cross-check methodology

The reference solution uses `softmax` + `argsort(descending=True, stable=True)` + `(sorted_p >= t).sum(-1)` to get
\(cnt\), writes the `keep` mask back with `scatter_`, and vectorizes the inverse transform with `cumsum` plus
`bool.argmax(-1)` (with an `any` fallback to the last position). The grading `_oracle` opens a fully independent
path: per row it converts probabilities into a Python list, obtains the descending order with
`sorted(range(V), key=lambda i: (-p[i], i))`, counts the contiguous prefix with a **for-else break** scan, zeroes
`keep` by index, normalizes with a hand-summed total, and does the inverse transform with a
`acc += x; if acc >= uv: break` loop (reusing the last element when `u_gate`/`u` has fewer entries than rows). The
cases cover 30 batches of \(\text{logits}\in\mathbb{R}^{5\times12}\) × \(\pi\in\{0,0.5,1\}\) ×
\(t\in\{0.01,0.05,0.3\}\) × \(m\in\{0,3,8\}\) with all three outputs cross-checked, plus five dedicated cases:
closed gate is identity, disabled parameters (including \(V=1\)), a single head candidate above the threshold means
no cut, `min_keep` blocks the cut, and the cut removes the head rather than the tail.

## Complexity

Per row: one softmax \(O(V)\), one descending sort \(O(V\log V)\) (down to \(O(V\cdot cnt)\) if only the prefix is
needed), prefix counting \(O(cnt)\le O(V)\), mask and normalization \(O(V)\), inverse-transform CDF \(O(V)\).
Space \(O(V)\) for `order`/`keep`/`cdf`. Upstream's `min_keep` and `probability` tests are constant time, so XTC is
a cheap link in the sampling chain.

## Practical notes

The scale of \(t\) is very sensitive: commonly \(0.05\sim0.2\). As \(t\) approaches 0.5 the cut only fires on the
rare steps where two head candidates are simultaneously very high; as \(t\) gets tiny (say \(10^{-3}\)) the passing
prefix spans dozens of ranks, the cut overshoots and the output visibly "steps one level down". \(\pi\) is the real
intensity knob — \(\pi=1\) makes every single step take the knife and repetition suppression turns into a stable
"preference for the runner-up"; \(\pi\approx0.3\) is closer to how upstream uses it. Give `min_keep` at least 2
explicitly: the \(V=2,\ t=0.5\) row measured in this batch shows that `min_keep=0` lets XTC collapse the
distribution to a single point, accidentally killing the randomness in low-temperature small-vocabulary settings.
Finally, when XTC is combined with `top-p`/`min-p` the order matters (upstream truncates the tail first, then XTC
cuts the head); this problem isolates it as a standalone operator, so the survivor count
\(V-\text{pos\_last}\) is measured against the **full vocabulary** rows, not against an already-truncated
candidate list.
