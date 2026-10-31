# KV Cache: ZSMerge Zero-Discard Weighted Merging

## Problem

Once the KV cache is full, the mainstream move is to **evict** tokens (H2O and SnapKV both do). ZSMerge (arXiv 2503.10714) goes the other way: nothing is ever discarded. For each incoming token it picks the lowest-scoring slot \(i\) in the **non-dense region**, merges it into the slot \(j\) in the **dense region** nearest to it, and lets the new token take over the vacated slot \(i\).

The merge is **count-weighted**. Every slot carries a `counts` value recording how many original tokens it averages. Folding \(i\) into \(j\) gives

\[
w_i=\text{counts}_i,\quad w_j=\text{counts}_j,\qquad
k_j\leftarrow\frac{w_i k_i+w_j k_j}{w_i+w_j},\quad
v_j\leftarrow\frac{w_i v_i+w_j v_j}{w_i+w_j}
\]

The weight is **inherited**: \(\text{counts}_j\leftarrow w_i+w_j\) and \(\text{counts}_i\leftarrow 1\). A chain of merges therefore equals one weighted average over the original tokens; a late token is not double-counted just because it was merged once.

Scores accumulate the same way: \(\text{scores}_j\leftarrow\text{scores}_i+\text{scores}_j\), and \(\text{scores}_i\leftarrow\) the new token's score.

The **dense region** is a positional prefix at the front of the cache (derived upstream from the `cache_dense` ratio); \(j\) may only be chosen there. Everything outside it is expendable. Upstream measures "nearest" by squared Euclidean distance (`metric="l2"`) or negated dot product (`metric="dot_product"`).

## Task

The inputs are a KV cache snapshot that already carries `counts` / `scores`, a dense mask, and one incoming token:

- `keys`, `values`: `(T, H, D)`, one shared dtype (float32 or float64);
- `scores`, `counts`: `(T, H)`, same dtype, all `counts` positive;
- `dense`: a `(T,)` bool mask; upstream it is always a positional prefix, which this task does not check;
- `new_key`, `new_value`: `(H, D)`; `new_score`: `(H,)`;
- `metric`: `"l2"` (default) or `"dot_product"`, affecting only the choice of \(j\).

Return a 5-key dict:

- `keys`, `values`: `(T, H, D)`, the whole cache after the write-back;
- `counts`, `scores`: `(T, H)`, likewise;
- `merged_slots`: `(H, 2)` int64, row \(h\) being `[i, j]`; when no merge happens \(j=-1\).

All four tensor outputs keep the input shape and dtype; `merged_slots` is always int64. Do the math in float64 and never mutate the inputs.

## Example

```
# T=5 slots, H=1 head, D=2
keys   = [[0, 0], [4, 4], [1, 0], [2, 2], [0, 1]]
values = 10 * keys
scores = [1, 2, -2, 5, 3]
counts = [1, 1, 3, 2, 1]
dense  = [1, 1, 0, 0, 0]
new_key, new_value, new_score = [3, 3], [30, 30], 1.5

i = 2                  # lowest score outside dense: min(-2, 5, 3)
j = 0                  # nearest dense key to k_i=[1, 0]: 1 vs 25 (squared L2)
w_i, w_j = 3, 1
keys[0]   = (3*[1, 0] + 1*[0, 0]) / 4   = [0.75, 0]
values[0] = (3*[10, 0] + 1*[0, 0]) / 4  = [7.5, 0]
counts[0] = 3 + 1 = 4
scores[0] = -2 + 1 = -1
keys[2], values[2], counts[2], scores[2] = [3, 3], [30, 30], 1, 1.5
merged_slots = [[2, 0]]
```

Here `l2` and `dot_product` genuinely pick different \(j\): against `[4, 4]` the negated dot product of \(k_i\) is \(-4\), below the \(0\) against `[0, 0]`, but the dense keys `[0, 0]` and `[4, 4]` score \(0\) and \(-16\), so `dot_product` selects \(j=1\). See the tests.

## Constraints

- **\(i\) comes only from the non-dense region**: the smallest `scores`, ties broken by the smaller slot index. A dense slot may never be sacrificed, however low its score.
- **\(j\) comes only from the dense region**: `l2` is the **squared** Euclidean distance (no square root, same ordering), `dot_product` is the **negated** dot product; ties again go to the smaller slot index.
- **Write \(j\) before \(i\)**: \(i\) is non-dense and \(j\) is dense, so \(i\neq j\) always and the two writes never collide.
- **All-False dense mask**: degenerates to "no merge", \(j=-1\), and the globally lowest-scoring slot is simply handed to the new token (upstream's `cache_dense == 0` branch).
- **All-True dense mask**: there is no expendable slot, so raise `ValueError`.
- **Determinism**: upstream relies on `torch.argmin` / `torch.topk`, whose tie order is unspecified; this task pins every tie to "smaller slot index".
