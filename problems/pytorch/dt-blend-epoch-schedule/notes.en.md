# Reference Approach: Four Gates, Each With One Trap Door

## Implementation

Four stages in series — each one's output is the next one's entire input. All five intermediate
results are returned, so a misreading anywhere surfaces in a downstream cell.

1. **Temperature weights.** `expo = 1.0 / T`, `e_i = w_i ** expo`, but a source with
   `sizes[i] == 0` is forced to zero **before** scaling. The order matters: scale first and the
   empty source still contributes to `total`, which depresses everyone else's probability and
   therefore their quota — while those sources had no extra documents to hand out anyway.
   Example 2 is exactly this: `sizes = [4, 0, 9]`, `weights = [1, 5, 4]`, `T = 2`, and source 1's
   \(\sqrt 5 = 2.236\dots\) has to disappear so that the remaining `1 : 2` becomes `1/3 : 2/3`.
2. **Normalisation.** Only a finite, strictly positive `total` can be normalised. All weights
   zero, every source empty, or only an empty source carrying weight — all three land here and
   raise `ValueError`.
3. **Largest-remainder quota.** `q = p * N`, `base = floor(q)`, `extra = N - sum(base)`, then the
   first `extra` indices in descending `(q - base)` and ascending index order get one each.
   **`extra <= k` holds by construction**: the gap between `base` and `q` is the sum of the
   fractional parts, each \(<1\), so `extra = N - sum(base) <= k`; and each index appears once in
   the ranking, so no source can be bumped twice. The tie rule is one stable sort, no hand-written
   loop.
4. **Smooth scheduling.** `slot_0 = j * N // c` is where the forward scan for a free slot starts.
   Integer division is used rather than `floor(j * N / c)`: the two are identical at these
   magnitudes (`j * N <= 4e10 < 2**53`, and `c <= N` keeps the quotient at least `1/c` away from
   an integer) but integer division does not depend on float rounding.
5. **Cyclic oversampling and global ids.** Emission `j` reads `j % sizes[i]`, so a quota above the
   document count re-reads the source in a ring; `offsets` is the exclusive prefix sum, so an
   empty source consumes no range and `flat` stays inside `[0, sum(sizes))`.

Can the forward scan of step 4 run off the end? The judge has no wrap branch because **it cannot
happen**: 5000 random quota vectors with `N <= 300` and `1..9` sources produced 0 overruns, and an
exhaustive enumeration of every integer split with `N <= 11`, `k <= 4` produced 0 as well. The
reason is that one source's ideal slots are always `N/c >= 1` apart, an earlier source can only
take its own slots, and the last source still finds room after its final emission.

What is the smooth layout worth? Take `probs = [0.4, 0.3, 0.2, 0.1]` with `N = 48` (quotas
`19/14/10/5`) and compare two schemes: the ideal slots of this problem, and a "one source at a
time" baseline that groups every sample of a source together. Gaps are measured **circularly**
(the wrap from the last slot to the first counts as one gap) and `cv` is their coefficient of
variation:

```text
# 4 sources, N = 48, quotas 19/14/10/5 (probs 0.4/0.3/0.2/0.1)
# max gap and gap coefficient of variation over the circular layout
  source 0: quota 19   smooth max gap  3  cv 0.198   | dump max gap 30  cv 2.563
  source 1: quota 14   smooth max gap  5  cv 0.286   | dump max gap 35  cv 2.554
  source 2: quota 10   smooth max gap  9  cv 0.435   | dump max gap 39  cv 2.375
  source 3: quota  5   smooth max gap 17  cv 0.565   | dump max gap 44  cv 1.792
```

Under the grouped baseline the 40% source vanishes from the last 30 steps of the epoch; ideal
slots push that number down to 3. A small source inevitably has a larger `cv` (five samples over
48 slots can only be 9 or 10 apart) — that is arithmetic, not an implementation artefact.

## Correctness checks

Twelve `check_*` functions, all compared against a structurally independent oracle: the oracle
computes `probs` with a left `reduce` after converting each entry to `float`, allocates the quota
by **repeatedly taking the largest remainder among the not-yet-bumped indices** (ties then fall to
the smaller index for free), schedules with a **union-find successor array** that jumps to the
first free slot while the ideal slot goes through a float `floor`, and derives global ids from
`sum(sizes[:src])` recomputed on the fly instead of a prebuilt table. The reference uses an
in-place `for` loop, one stable sort, a linear forward scan with integer division, and a
precomputed `offsets`. No intermediate result is shared. Beyond the 5-tuple comparison, `_verify`
enforces: every `probs` entry is a `float` and the sum is within \(10^{-9}\) of 1;
`sum(counts) == num_samples`; `floor(q_i) <= counts_i <= floor(q_i) + 1`;
`counts[i] == 0` whenever `sizes[i] == 0`; source `i` appears exactly `counts[i]` times in a
`schedule` as long as the budget; `0 <= doc_indices[t] < sizes[schedule[t]]`; decoding
`flat_indices[t]` with `bisect` must reproduce both `(source, doc)`; and `sizes` and `weights` are
unmutated.

- `check_example`: hand-computed 5-tuples for all four statement examples (the third one also
  pins the `j % 5` ring).
- `check_matches_oracle`: 27 fixtures covering a single source, an empty source first/middle/last,
  zero weight, zero budget, a budget below the source count, equal weights, `T = 0.5/1/2/3/4/100`,
  million-record sources, and `[1e6, 1e6]` where the quota vastly exceeds the document count.
- `check_quota_is_largest_remainder`: with `q = [0.5, 0.5], N = 1` rounding (banker's) yields
  `[0, 0]` while largest remainder yields `[1, 0]`; with `q = [1.5, 1.5], N = 3` per-entry
  rounding overshoots to `[2, 2]`; with `q = [7/3] * 3, N = 7` flooring every entry falls short
  at `[2, 2, 2]`.
- `check_temperature_flattens_and_sharpens`: `T = 1` must reproduce the raw weight ratio bit for
  bit (`1/7, 2/7, 4/7`); `T = 0.5` must square it (`1/21, 4/21, 16/21`) with quota `[1, 3, 10]`;
  `T = 2` gives `[3, 5, 6]`; `T = 1e6` drives two sources to `[6, 7]`, nearly even.
- `check_cyclic_oversampling`: quotas 5/4 over 2/2 documents must read `[0, 1, 0, 1, 0]` and
  `[0, 1, 0, 1]`; a single source emits document 0 four times; 25 emissions of source 0 form a
  13-entry `0,1,2` ring.
- `check_placement_is_by_source_order`: `counts = [2, 3]` and `[2, 3, 1]`, where this problem's
  ascending-index placement differs from descending-quota placement — both are compared and the
  inequality is asserted explicitly.
- `check_first_source_sits_on_ideal_slots`: for every fixture, the smallest index with a positive
  quota must sit **exactly** on `[floor(j*N/c)]` — it is placed first into an empty array, so this
  can be asserted precisely.
- `check_zero_size_source_is_never_used`: `[0, 5, 0, 5]` where the empty sources carry the larger
  weights still yields `counts = [0, 3, 0, 3]`, `set(schedule) == {1, 3}` and a `flat` list that
  skips the empty ranges; on `[4, 0, 6]` an empty source with weight 3 also earns nothing.
- `check_flat_indices_are_global_ids`: hand-computed `flat` for `sizes = [3, 1, 4]`, `N = 8`.
- `check_num_samples_zero_and_tiny`: a zero budget still normalises `probs` and returns three empty
  lists; a budget of 2 below the source count 4 emits only `[1, 1, 0, 0]`.
- `check_scale`: 40 sources (two empty, one with zero weight) across four `(N, T)` pairs, plus
  `[10**5] * 3` with `N = 30000` — three interleaved ladders (`schedule[:6] == [0,1,2,0,1,2]`) and
  source 0's documents being exactly `range(10000)`.
- `check_invalid`: 39 rejected inputs plus 4 legal boundaries. Four corners deserve a mention: a
  `10**400` weight raises `OverflowError` inside `float()`, `1e200 ** 2` raises `OverflowError`
  too (both must become `ValueError`), `[1e308, 1e308]` overflows the sum to `inf`, and
  `(-2.0) ** 0.5` **returns a complex number** in Python instead of raising — which is why a
  negative weight has to be rejected before any scaling happens.

Mutation testing ran 24 mutants and all 24 were caught: temperature ignored, exponent inverted,
empty source not masked, `base` rounded or ceiled, ties to the larger index, `extra` handed to the
first indices, quota left as floats, placement by descending quota, ideal slot rounded or pinned
to 0, backward scan, no scan at all, document number taken from the slot or left un-modded, `flat`
dropping the offset, `offsets` made inclusive, `bool` accepted as an `int`, negative weights
allowed, the finiteness of `total` unchecked, zero temperature allowed, fractional budgets allowed,
and `probs` hard-coded to `1/k`.

Two further rewrites are **equivalent** and do not count: summing `total` backwards, and writing
`e/total` as `e * (1/total)`. They differ from the reference only in the last bit, the statement
compares `probs` with a 1e-12 tolerance, and no fixture flips because of them — failing a candidate
over that would be gratuitous, which is why the statement says "`total` is the sum of the entries
of `e`" instead of "accumulated left-to-right".

## Scope and sources

- **No attribution claimed.** DoReMi / megatron mixtures and HuggingFace's blending scripts all use
  \(w^{1/T}\) with an integer quota, but there is no implementation to compare against offline
  (`.venv-eval` has only torch / numpy / transformers — no datasets, datatrove or megatron).
  `expo = 1/T`, forcing empty sources to zero, "largest remainder with ties to the smaller index",
  "place in ascending index order and scan forward from the ideal slot", and "`j % sizes` cycling"
  are all **defined by this problem**. Real pipelines differ at every one of these points (samples
  versus tokens as the unit, whether a document may repeat, whether each source is shuffled
  internally).
- **The properties are backed by a local measurement.** For the temperature gate, two
  monotonicities should hold — higher temperature means higher entropy, and fewer sources starve.
  20 random weight vectors over 6 sources, `N = 100`, natural log:

  ```text
  # 20 random weight vectors over 6 sources, N = 100, sizes = 1000 each
  # H = natural-log entropy of probs (uniform would be ln 6 = 1.7918)
  # starved = entries with counts == 0 while probs * N > 0, summed over the 20 blends
     T      mean H(probs)   max |counts - q|   starved sources   max |sum probs - 1|
     0.25        0.9967           0.635099                35            2.22e-16
     0.50        1.3482           0.584094                11            2.22e-16
     1.00        1.6032           0.650223                 1            2.22e-16
     2.00        1.7265           0.585846                 0            2.22e-16
     4.00        1.7722           0.676399                 0            2.22e-16
    16.00        1.7904           0.750774                 0            2.22e-16

  # ideal-slot forward scan, random quota vectors (N <= 300, 1..9 sources)
  5000 vectors scanned, overruns beyond the array: 0
  ```

  Entropy rises monotonically with temperature and approaches `ln 6`; at `T = 0.25`, 35 of the 120
  "source times blend" pairs are squeezed to a zero quota — the quantitative version of "normalising
  raw weights starves the small language", which is the whole reason temperature exists.
  `max |counts - q|` stays below 1 throughout, matching the construction (it does not shrink with
  temperature, because it is the intrinsic error of an integer allocation).
- **Division of labour with neighbours**: `dt-rank-shard-indices` (549) is about how one global
  index is cut across ranks (strided slices, head padding, the global coverage account); this
  problem is about what happens **before** the cut — how many frames each source occupies and
  where they sit. `sequence-pack-pad` (120) covers fixed-length packing.
  `fw-distributed-sampler` (253) keeps the index account of a single-machine `DistributedSampler`
  and has no multi-source quota. `dt-unigram-sample` (462) samples a Unigram posterior path inside
  a tokenizer, which is unrelated to data blending — it samples tokens, this one samples documents.
