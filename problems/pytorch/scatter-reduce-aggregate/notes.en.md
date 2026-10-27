# In Depth: The CPU Semantics of scatter_reduce Aggregated Writes

## Source

- The operator being reproduced: `torch.Tensor.scatter_reduce_(dim, index, src, reduce, include_self=True)` (stable API since PyTorch 1.12, reduce ∈ `sum|prod|mean|amax|amin`). This problem matches the **actual numeric behaviour of the CPU implementation**, not a paraphrase of the docs.
- Welford incremental mean: the Welford (1962) / Chan-Lovell-Martin family of "mergeable statistics"; the CPU kernel updates `mean` serially as `m += (v - m)/n` instead of summing first and dividing.
- Application context: MoE unpermute (aggregating expert outputs back into token order by routing counts), the scattered-write form of `segment_reduce`, writing variable-length sequence statistics back into fixed-length tables. All of these need a **deterministic order** when several contributions hit one target — which is exactly the hard part here.

## Full derivation and index convention

**Indices and linear addresses**: let input have shape $(S_0,\dots,S_{n-1})$ with row-major strides $\sigma_j=\prod_{k>j}S_k$. A flat index $t$ of index decodes to coordinates $c=(\ell_0,\dots,\ell_{n-1})$ (successive `//` and `%` from $n-1$ down to 0), and the target address is
\[m(t)=\sum_{j\ne d}\ell_j\sigma_j+\text{index}[t]\cdot\sigma_d .\]
**The dim coordinate is replaced by the index value while all other coordinates come from index itself** — this is what makes "index pairs with src by coordinate, and src may be larger" work: surplus src elements have no corresponding index coordinate, so they simply do not participate.

**Why contributions arrive in "ascending dim coordinate"**: $t$ increases in row-major order, and within one pos the coordinate $c_d$ is exactly what varies (same pos means all other coordinates and the index value are equal), which row-major order advances fastest. So "row-major arrival order" and "ascending dim coordinate" are two phrasings of one thing; the reference groups with `argsort(pos, stable=True)`, and stability guarantees the in-group order is row-major.

**mean has two independently observable deviations.** The first is **counting**: with include_self=True the seed is $m=\text{original value}$ and $n=1$, so the original value enters both numerator and denominator. Measured: base=0 with contributions $[1,2,3,4]$ → native **2.0**, whereas "average the scatter values only" gives 2.5; base=0 with contributions $[10^8,1,1,1]$ → native **\(2\times10^7\)**, the same mistake gives \(2.5\times10^7\). The second is **serial rounding**: $m\leftarrow m+(v-m)/n$ rounds at every step, which diverges wildly from "sum then divide" across magnitudes. Measured with base=0 and contributions $[10^8,1,1,-10^8]$: the serial path runs $5{\times}10^7\to3.3333{\times}10^7\to2.5{\times}10^7\to0$ (the last step cancels exactly in float32) giving **0.0**, bit-identical to native, while summing/counting gives **0.4** (under exact arithmetic both would be 0.4). In short: **writing the Welford formula is not enough — it must run serially in row-major order, rounding inside the input dtype**, which is why the reference updates scalar tensors one at a time instead of vectorized `sum/count`.

**The seed semantics of include_self=False**: a touched pos is seeded with **its first scatter value**, then folds the rest (mean's $n$ also starts at 1). So "whether the original value participates" only affects **touched** positions; untouched ones keep the original under both settings. Three observable spots: prod (the $5,7$ vs $6,4$ contrast — the input values 5/7 never enter the result), sum ($10{+}1{+}2$ vs $1{+}2$), mean (base $10^8$ with contributions $[1,1,1,1]$: inc=True → \(2\times10^7\), inc=False → **1.0**).

**amax/amin "competing with the original value"**: with inc=True the accumulator and the input value both compete; with inc=False only scatter values do. In the example amax/inc=True returns exactly the input (every scatter value is smaller) while the two amin columns coincide (the first scatter value already beats the original). Together these two rows pin the notion of "competing set": an implementation that just writes `torch.maximum(out, src)` cannot satisfy both.

**Empty index**: with `index.numel()==0` the function returns a **copy** of input (asserted with `torch.equal`). A copy, not the same object — the judge also snapshots input to prove it is unmodified, so cloning is mandatory.

**Why the error surface is deliberately split**: an illegal `reduce` raises **ValueError** (a Python-level value-domain error), while out-of-bounds index, dimension-count mismatch and dtype mismatch raise **RuntimeError** and an out-of-range dim raises **IndexError** (matching the native tensor method). This is fixed by the task statement; do not "unify" the native ones into ValueError.

## Key points and common mistakes

1. **The original value occupies a slot in mean**: averaging only the scatter values is off by 0.5 and by \(5\times10^6\) in the two base=0 cases; summing then dividing in float64 diverges from native wherever magnitudes are extreme (see the measured $[10^8,1,1,-10^8]$ case above).
2. **include_self=False seeds with the first contribution**, not 0 (0 would annihilate sum/prod results) and not "skip the position".
3. **Untouched positions keep the input value** under both settings — returning 0 there for `mean/inc=False` is the classic error.
4. **Contribution order is row-major** (equivalently ascending dim coordinate): a parallel reduction (e.g. `index_add_` then divide by counts) will not match in floating point.
5. **When src is larger than index, take the same-coordinate prefix**, not the first `index.numel()` flat elements (the two differ once shapes are not identical).
6. **Only the dim axis may be wider than input**; every other axis must satisfy `<=` — writing the constraint as "all axes ≤ input" wrongly rejects legal inputs.
7. **Normalize negative dims first**, then range-check (outside $[-n,n)$ → IndexError).
8. **dtype checks**: src dtype ≠ input dtype → RuntimeError; index must be an integer tensor.
9. **Never modify input**: write into a clone; the `p` in `out[p] = ...` is a **flat** address, so reshape back to `input.shape` at the end.
10. **Out-of-range includes negatives**: a negative index is also a RuntimeError natively (Python's negative-index semantics do **not** apply to index tensors), so validate $[0,\text{input.size}(d))$ before writing.

## Differences from neighboring problems

- **vs `gather-scatter`**: that problem implements plain `scatter_` (an **overwrite**, where repeated coordinates mean "last writer wins", no aggregation) and `gather_` (reading); there is no reduce parameter and no include_self. `scatter_add_` is merely the `reduce="sum"` always-with-self special case. Everything hard here is "how to merge repeated coordinates".
- **vs `distributed-reduce-scatter`**: that is a **cross-rank** collective — each rank contributes a same-shaped shard, the results are reduced and scattered back; here it is a single-tensor write where target positions may repeat. Similar name, different mechanism.
- **vs `searchsorted-bincount` (same batch)**: `bincount` is exactly the one-dimensional special case `reduce="sum"` with src≡1 and include_self=False; this problem generalizes it to any number of dimensions, five reductions and both include_self states.
- **vs `tensor-shape-indexing`**: that problem is about shape inference and index semantics (the read side); this one is about address computation with aggregation on the write side.

## Cross-check convention and complexity

- **The primary oracle is the native `Tensor.scatter_reduce_` itself**: 6 seeds × 5 reductions × 2 include_self × three data groups for dim{0,1,-1}, plus a 3-D case (index narrower than input along dim) and the "src larger than index" branch; float32 \(10^{-5}\), float64 \(10^{-9}\). Using the library as oracle is legitimate precisely because the task is to **reproduce** this CPU semantics bit-for-bit, rounding included, not to invent a convention.
- A **second, independent Python oracle** `_oracle_loop` covers sum/amax/mean × 2 include_self × dim{0,1}: it walks coordinates with `itertools.product` and keeps `(acc, n)` state in a dict — a completely different path from "compute flat positions + stable-sort grouping", so it catches position-arithmetic errors such as forgetting to substitute the dim coordinate.
- Discriminating cases: the two extreme mean examples (with the batch-formula counter-values), prod's first-contribution seeding, mean's counting and untouched-keeps-original, "all five reductions are mutually `torch.equal` when every position receives exactly one contribution" (aggregation degenerates to assignment, which also validates the include_self=False seed), prod over a zero-valued input, empty index, and the three error paths (illegal reduce, out-of-range index, out-of-range dim).
- All ten cells of the worked example were backfilled from the reference run and most are checkable on paper (see the step-by-step arithmetic in the statement).
- Complexity: with $T$ = number of index elements, flat positions cost $O(T)$, the stable sort $O(T\log T)$, and the grouped serial fold $O(T)$; memory is $O(T)$ plus a copy of input. The native kernel uses atomics/segmented parallelism; reproducing its **serial rounding** forces the sorting and serial fold costs — the typical price of "reproduce the semantics" problems.
