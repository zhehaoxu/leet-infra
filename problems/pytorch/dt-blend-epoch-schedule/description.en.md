# Dataset Blending: Temperature Weights, Largest-Remainder Quota and Smooth Scheduling

## Task

"How many samples does each source contribute?" is a recipe that has to be designed, not inherited. You have `k` sources, source `i` holds `sizes[i]` records and carries a raw weight `weights[i]`, and the epoch budget is `num_samples` samples in total. Turning that recipe into a **deterministic** index stream passes four gates:

1. **Temperature blending.** Normalising the raw weights directly is usually too steep — a small language barely shows up. The standard fix is to normalise \(w_i^{1/T}\) instead: \(T>1\) flattens the mixture towards uniform, \(T<1\) sharpens it towards the heaviest source, and \(T=1\) reproduces the raw ratio. This is the alpha in DoReMi / megatron-style blends.
2. **Integer quota.** Probability times budget is rarely an integer. Flooring every entry loses a few samples, rounding every entry can overshoot the budget. The **largest-remainder** rule (Hare quota) is the allocation that simultaneously gives "total exactly equals the budget" and "each entry is within 1 of its ideal value" — and the tie-break for the last few seats must be spelled out.
3. **Smooth scheduling.** A quota says how many, not where. Dumping one source at the head of the epoch leaves the tail dominated by a single domain. The smooth rule computes an ideal slot for emission `j` and then walks forward to the first free one, so the layout is a pure function — reproducible and shardable.
4. **Cyclic oversampling.** A source's quota can exceed its record count (unavoidable once `num_samples` exceeds the corpus), and then emission `j` reads document `j % sizes[i]` — round-robin re-reading rather than drawing without replacement. A source with zero records must get no quota at all.

This problem runs all four gates as one pipeline and asks for **all five intermediate results**: probabilities, quotas, the source stream, per-source document numbers, and global document numbers. Any misreading shows up in a downstream cell. Pure Python float and integer arithmetic throughout — no RNG, no `random.shuffle`, no `WeightedRandomSampler`.

## Input

- `sizes`: non-empty `list[int]`, records per source, `0 <= sizes[i]`.
- `weights`: `list` of the same length, each entry a non-negative `int`/`float` that converts to a finite `float`.
- `num_samples`: non-negative integer, the sample budget for this epoch.
- `temperature`: a positive `int`/`float` that converts to a finite `float`, the blending temperature.

**Forbidden**: any randomness, any sampler library. All four gates are hand-written.

## Exact Rules

Let `k = len(sizes)`. The order below is part of the contract.

1. **Temperature weights.** With `expo = 1.0 / temperature`,
   \[ e_i = \begin{cases} 0.0, & sizes[i] = 0\\ w_i^{\,expo}, & \text{otherwise}\end{cases} \]
   a source with zero records is **forced** to zero regardless of its weight.
2. **Normalisation.** `total` is the sum of the entries of `e`, and `probs[i] = e_i / total` (floats are compared with a 1e-12 tolerance, so the association order does not matter). `total` must be finite and \(> 0\), otherwise raise `ValueError`.
3. **Largest-remainder quota.** Ideal value \(q_i = probs[i] \cdot num\_samples\), `base_i = floor(q_i)` (floor, not round). Let `extra = num_samples - sum(base)`; sort the indices by \((q_i - base_i)\) **descending**, **ties broken by the smaller index first**, and add 1 to each of the first `extra` sources. The resulting integer `counts` satisfy `sum(counts) == num_samples` and \(|\,counts_i - q_i\,| < 1\).
4. **Smooth scheduling.** Start with an empty sequence of length `num_samples`. Place sources in **ascending index order**; emission `j` of a source (its own counter, `0 <= j < counts[i]`) has ideal slot
   \[ slot_0 = \left\lfloor \frac{j \cdot num\_samples}{counts[i]} \right\rfloor \]
   and scans **forward** from `slot_0` to the first free slot, where `schedule[slot] = i` is written. A source with `counts[i] = 0` occupies nothing. A free slot always exists within bounds.
5. **Cyclic oversampling and global ids.** If emission `j` of a source lands on slot `t`, then `doc_indices[t] = j % sizes[i]`. `offsets` is the exclusive prefix sum of `sizes` (`offsets[0] = 0`, `offsets[i] = sizes[0] + ... + sizes[i-1]`), and `flat_indices[t] = offsets[schedule[t]] + doc_indices[t]`, i.e. the index of that document in the concatenated corpus.

## Return

The 5-tuple `(probs, counts, schedule, doc_indices, flat_indices)` of plain Python built-ins:

- `probs`: `list[float]` of length `k`, summing to 1 up to float error.
- `counts`: `list[int]` of length `k`, summing to `num_samples`.
- `schedule`: `list[int]` of length `num_samples`; value `i` appears exactly `counts[i]` times.
- `doc_indices`: `list[int]` of length `num_samples`, the **within-source** document number, `0 <= doc_indices[t] < sizes[schedule[t]]`.
- `flat_indices`: `list[int]` of length `num_samples`, the global document number.

Any illegal argument raises `ValueError`: `sizes` not a list or empty, an element not an `int` (`bool` is not an `int`) or negative; `weights` not a list, length different from `k`, an element not an `int`/`float` (`bool` does not count), negative, `NaN`, `inf`, or too large to convert to a `float`; `num_samples` not an `int` (`bool` does not count) or negative; `temperature` not an `int`/`float` (`bool` does not count), `<= 0`, `NaN`, `inf`, or too large to convert to a `float`; the power overflowing, `total` non-finite, or `total == 0` (every weight zero, or every non-empty source carrying weight 0). `num_samples = 0` is legal: `probs` is still normalised and the last three lists are empty. `sizes` and `weights` must not be mutated.

## Examples

Example 1, temperature 1 means the raw ratio, and the gap is closed by largest remainder:

```
sizes = [2, 3, 5], weights = [1, 1, 2], num_samples = 7, temperature = 1.0

expo = 1/1 = 1, e = [1, 1, 2], total = 4
probs  = [0.25, 0.25, 0.5]
q      = [1.75, 1.75, 3.5]          base = [1, 1, 3]  sum 5, extra 2
rem    = [0.75, 0.75, 0.5]          descending, tie -> smaller index first
                                    -> bump sources 0 and 1
counts = [2, 2, 3]

# place source 0 (c=2) first: ideals floor(0*7/2)=0, floor(1*7/2)=3
# then source 1 (c=2): ideal 0 busy -> 1 ; ideal 3 busy -> 4
# then source 2 (c=3): ideal 0 busy -> 2 ; ideal 2 busy -> 3 busy -> 5 ; ideal 4 busy -> 6
schedule    = [0, 1, 2, 0, 1, 2, 2]
doc_indices = [0, 0, 0, 1, 1, 1, 2]      # per-source emission counter j
offsets     = [0, 2, 5]                  # exclusive prefix sum of sizes
flat        = [0, 2, 5, 1, 3, 6, 7]
```

Example 2, a zero-record source is dropped outright and temperature 2 takes a square root:

```
sizes = [4, 0, 9], weights = [1, 5, 4], num_samples = 5, temperature = 2.0

expo = 0.5, e = [1.0, 0.0, 2.0]     # source 1 has 0 documents -> forced to 0
                                     # (sqrt(5) = 2.236... would have been its entry)
probs  = [0.3333333333333333, 0.0, 0.6666666666666666]
q      = [1.6666666666666665, 0.0, 3.333333333333333]
counts = [2, 0, 3]                  # base [1,0,3], extra 1 goes to the largest remainder

schedule    = [0, 2, 0, 2, 2]
doc_indices = [0, 0, 1, 1, 2]
offsets     = [0, 4, 4]             # the empty source still consumes no range
flat        = [0, 4, 1, 5, 6]
```

Example 3, temperature 0.5 squares the gap, quota 7 exceeds the 5 available records, so reading wraps:

```
sizes = [5, 5], weights = [1, 3], num_samples = 8, temperature = 0.5

expo = 2.0, e = [1.0, 9.0], probs = [0.1, 0.9]
q = [0.8, 7.2], base = [0, 7], extra = 1 -> remainder 0.8 wins
counts = [1, 7]

schedule    = [0, 1, 1, 1, 1, 1, 1, 1]
doc_indices = [0, 0, 1, 2, 3, 4, 0, 1]   # j % 5 wraps: source 1 is oversampled
offsets     = [0, 5]
flat        = [0, 5, 6, 7, 8, 9, 5, 6]   # ids 5 and 6 appear twice, by design
```

One more look at the tie-break: four equally weighted sources with budget 6 make every ideal
value `1.5`, so `base` is all ones and `extra` is 2; both seats go by "smaller index first",
hence `counts == [2, 2, 1, 1]` and **not** `[1, 1, 2, 2]`.
