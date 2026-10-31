# Multimodal: LLaVA-PruMerge IQR-Adaptive Token Count

## Task

LLaVA-PruMerge (Shang et al., arXiv:2403.15388) shrinks visual tokens inside the CLIP vision tower. The sibling problem (order 536) pins "how many tokens to keep" to an explicit `reduction_ratio` argument; this one adds the other half — the **official adaptive ratio**: a Tukey 1.5× IQR outlier rule counts outliers among the attention values to derive the kept ratio, then the key patches are selected accordingly.

This task covers only "adaptive ratio + selection": patch features, the real forward pass and the merge are all out of scope; `cls_attn` has already been averaged over heads by the caller. Parameter, shape and exception conventions live in the bilingual contract in the starter; the rules below define the computation only.

## Rules

### Step one: the adaptive ratio (a line-by-line transcription of upstream `outlier_dectection`)

1. Flatten the **whole `[B,N]`** into one sequence — the ratio is shared across the batch, not computed per row — and cast it to **float32 first** (float64 inputs lose precision here; this is an upstream behaviour point with a dedicated boundary test).
2. `Q1` and `Q3` are the 25th and 75th percentiles under numpy's default **linear interpolation**.
3. `IQR = Q3 - Q1` and `upper = Q3 + 1.5 * IQR` (Tukey 1.5, upper tail only; the commented-out lower bound is out of scope).
4. `ratio = (# elements STRICTLY greater than upper) / (sequence length)`, a Python float.

### Step two: selection (same conventions as order 536)

- The kept count is `k = int(N * ratio)`, truncating.
- **The key difference from 536**: `k == 0` is a **legal result** here — the adaptive path may keep nothing (`torch.topk(..., 0)` returns an empty set), in which case `kept` is an empty `[B,0]` tensor and `pruned` is the full `0..N-1`; 536 raises `ValueError` in the same situation.
- `kept` holds the `k` largest `cls_attn` indices per row, ties resolved as "value descending, index ascending", emitted in **ascending index order**; `pruned` is the ascending complement.

### Mathematical guarantee (asserted by the tests)

At least one element of any finite sequence sits near `Q1` and is not strictly greater than `upper`, so `ratio < 1` always holds, `k ≤ N-1`, and `pruned` is never empty.

## Example 1: constant values → zero ratio

`B=2, N=5`, every element `0.1` (float32):

```text
ratio = 0.0
kept_indexs   = shape [2, 0] (empty)
pruned_indexs = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
```

With identical values `Q1 = Q3`, `IQR = 0`, `upper = Q3`, and "strictly greater" admits nothing equal to the bound.

## Example 2: a single outlier

`B=1, N=8` (float64), `cls_attn = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.50]`:

```text
ratio = 0.125            # 1 / 8
kept_indexs   = [[7]]
pruned_indexs = [[0, 1, 2, 3, 4, 5, 6]]
```

`Q1 ≈ 0.0275`, `Q3 ≈ 0.0625`, `upper ≈ 0.115`; only `0.50` crosses the bound and `k = int(8 * 0.125) = 1`.

## Example 3: the ratio flattens across the batch

`B=2, N=4` (float32), row 1 all `0.1`, row 2 `[0.1, 0.1, 0.1, 0.9]`:

```text
ratio = 0.125            # exactly 1 of 8 values crosses, shared by both rows
kept_indexs   = shape [2, 0]   # k = int(4 * 0.125) = 0, neither row keeps anything
pruned_indexs = [[0, 1, 2, 3], [0, 1, 2, 3]]
```

A per-row ratio would give row 2 `0.25` and `k = 1` — this is the witness against the per-row mutation.

## Judging

10 `check_*` functions: hand-computed anchors (zero ratio, single outlier, flattened sharing, tie selection), the `k=0` empty selection, float32/float64 input consistency (cast happens first), a float32-cast boundary case (a `nextafter` step is an outlier in float64 but merges after the cast), noncontiguous views and input immutability, mixed-distribution random cases against an independent oracle, and all invalid inputs. The oracle reuses no numpy/torch percentile or selection: the percentile is transcribed in pure Python with float32 step arithmetic plus numpy's two-sided lerp (verified bit-exact against `np.percentile` over 40k random cases), and selection is scalar loops.

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-prumerge-iqr-adaptive --mode submit
```
