# ToMe's bipartite soft matching merge ledger: exact cosine comparison, token merging, and size-weighted normalization

## Background

Token Merging (ToMe) does not delete tokens — it **merges similar tokens into one**: each token finds its "merge target", and the merged-away token is absorbed by that target. The token count drops while each surviving token carries more information.

The official `facebookresearch/ToMe` implementation uses **bipartite soft matching**: the token sequence is split **alternately by parity** into `a` (even positions) and `b` (odd positions); only `r` tokens from `a` are selected to disappear (`src`), each `src` finds its most similar `b` (`dst`), and merging lets `dst` absorb `src`. The remaining `unm` (unmerged `a` tokens) concatenated with `b` forms the next layer's token sequence.

Several details in this path are very easy to get wrong:

- **`-inf` protection lives on two different axes**: `class_token` protects **row 0** (the first element on the `a` side), `distill_token` protects **column 0** (the first element on the `b` side). Writing both on the same axis is entirely wrong.
- **`scatter_reduce`'s `mean` denominator includes the target's own value** (`include_self=True`), not just the number of incoming `src` tokens.
- **`merge_wavg`'s denominator is the sum of sizes**, not the token count.
- **The official implementation is floating point**, and where two cosines are *exactly* equal, a 1e-17 rounding difference can flip the choice made by `max` and `argsort` — the same input may yield a different merge plan on different machines.

This problem reconstructs the **pure numerical** ledger of that path. To keep the ledger decidable, the inputs are restricted to integers and the similarity comparison is pinned to an **exact rational** form (no square roots, integer cross-multiplication), with the tie rules pinned as well. ViT forward passes, GPU kernels, and the actual `timm` patch wiring are out of scope.

## General conventions

- All inputs are nested lists of Python `int` / `float`. No tensors, gradients, or GPU.
- `metric` has shape `[B][T][C]` (batch × token × channel), its elements are **integers**, and every row has **nonzero** norm (otherwise the cosine is undefined).
- `x` has shape `[B][T][C]` with integer elements; `size` has shape `[B][T][1]` with positive integer elements.
- Indices are 0-based and **inclusive**.
- `T` denotes the token count, `C` the channel count, `B` the batch count.
- Return values are always a `dict`; the field names are given per mode below.
- All modes are invoked through the single entry point `solve(mode, **kw)`.

## Exact cosine comparison (the core convention of this problem)

The official code computes `metric / metric.norm(dim=-1)` followed by `a @ b.T` to obtain cosine similarity. This problem replaces it with an **exact rational** convention:

```
s_ij   = Σ_c a_i[c] * b_j[c]        # exact integer dot product
m2_i   = Σ_c a_i[c]^2                # exact integer squared norm
cos_ij = s_ij / sqrt(m2_i * m2_j)   # the true value
```

When comparing two candidates `u = cos_ij` and `v = cos_pq`, **no square root is taken**; integer cross-multiplication is used instead:

```
sign first: non-negative > negative
both non-negative: u >= v  <=>  s_ij^2 * m2_p * m2_q >= s_pq^2 * m2_i * m2_j
both negative:     u >= v  <=>  s_ij^2 * m2_p * m2_q <= s_pq^2 * m2_i * m2_j
```

**Tie rules** (undefined in the official floating point code, pinned here):

- When `max` encounters tied maximal values, the **smallest index** is chosen;
- Sorting by descending similarity is treated as a **stable** sort, so ties keep the original ascending index order.

Positions protected by `-inf` (see below) behave as negative infinity: they sort last and can never be chosen as a `dst`.

## Modes and rules

### 1. `match_indices`: the index computation of bipartite soft matching

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `metric` | 3-D list `[B][T][C]` of integers | required |
| `r` | number of tokens to merge away | required |
| `class_token` | whether a class token exists | `False` |
| `distill_token` | whether a distillation token exists | `False` |

Steps:

```
# 1. Effective r: at most half (minus the protected tokens)
protected = int(class_token) + int(distill_token)
r_effective = min(r, (T - protected) // 2)

# 2. Alternating split: a side is even positions, b side is odd positions
t1 = (T + 1) // 2      # length of the a side
t2 = T // 2            # length of the b side
a = metric[:, 0::2, :] # even positions
b = metric[:, 1::2, :] # odd positions

# 3. Similarity matrix (exact convention); zero-norm rows/columns are -inf
scores[i][j] = cos(a_i, b_j)

# 4. -inf protection on two different axes
if class_token:
    scores[0][:] = -inf        # entire row 0
if distill_token:
    scores[:][0] = -inf        # entire column 0

# 5. Each a element picks its most similar b (ties -> smallest index)
node_max[i], node_idx[i] = max_j scores[i][j]

# 6. Stable descending sort by node_max; the first r_effective become src
order = stable_argsort_desc(node_max)
src_idx = order[:r_effective]
unm_idx = order[r_effective:]
dst_idx = [node_idx[i] for i in src_idx]

# 7. Only class_token sorts unm_idx (to keep the class token first)
if class_token:
    unm_idx = sorted(unm_idx)
```

**The `r_effective <= 0` special case**: the official code returns `do_nothing` (no matching at all, no merging). This problem's convention:

- `src_idx` / `dst_idx` are empty lists;
- `unm_idx` is `[0, 1, ..., t1-1]` (all `a`-side indices, ascending);
- `node_max` is an empty list (not computed).

Returned fields:

- `r_effective`: the effective `r`;
- `src_idx`: length `B`; entry `b` is a list of `r_effective` **a-side indices** ordered by descending similarity;
- `dst_idx`: length `B`; entry `b` is a list of `r_effective` **b-side indices** (each `src`'s match target);
- `unm_idx`: length `B`; entry `b` lists the unmerged a-side indices;
- `node_max`: length `B`; entry `b` has length `t1`, where item `i` is the cosine between `a_i` and its best match (`-inf` when protected or of zero norm);
- `num_src`: `t1`, the a-side length;
- `num_dst`: `t2`, the b-side length.

Pitfalls:

- **Do not mix up the protection axes.** `class_token` → `scores[0][:] = -inf` (**row**); `distill_token` → `scores[:][0] = -inf` (**column**). Clearing `scores[0][:]` means `a_0` cannot be a match *source* for any `b` (its `node_max` is `-inf` and it sorts last); clearing column 0 means `b_0` cannot be chosen by any `a`.
- **With both `class_token` and `distill_token`, `protected = 2`**, so the cap on `r_effective` is `(T - 2) // 2`, stricter than with only one of them.
- **Only `class_token` triggers the `unm_idx` sort.** With `distill_token` alone, `unm_idx` stays in descending similarity order. This "sort only under class_token" condition is easy to invert.
- **`dst_idx` may contain duplicates**: several `src` tokens may match the same `b` (their most similar `b` is identical). That is official behaviour, not a bug.
- **`node_max` holds cosines**, not dot products and not normalized vectors; protected rows hold `-inf`.
- **A negative `r` yields a negative `r_effective`** (`min` does not clamp to 0); the judge returns the value as-is.
- **For odd `T` the `a` side is one longer than the `b` side** (`t1 = (T+1)//2`, `t2 = T//2`), and the split follows the **even/odd positions of the original sequence**, not "first half / second half".

### 2. `merge_reduce`: merging (gather + scatter_reduce + output reordering)

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | 3-D list `[B][T][C]` | required |
| `src_idx` | merged-away a-side indices (length `B`) | required |
| `dst_idx` | b-side target of each `src` (length `B`) | required |
| `unm_idx` | unmerged a-side indices (length `B`) | required |
| `reduce` | `"mean"` / `"sum"` / `"amax"` | `"mean"` |
| `distill_token` | whether to use the interleaved output layout | `False` |

Steps:

```
# 1. Alternating split
src_rows = x[:, 0::2, :]    # a side
dst_rows = x[:, 1::2, :]    # b side

# 2. gather
unm = src_rows.gather(unm_idx)     # unmerged a-side tokens
src = src_rows.gather(src_idx)     # merged-away a-side tokens

# 3. scatter_reduce onto the b side (include_self=True, the official default)
dst = dst_rows.scatter_reduce(-2, dst_idx, src, reduce=reduce)

# 4. Output reordering
if distill_token:
    merged = cat([unm[:1], dst[:1], unm[1:], dst[1:]])
else:
    merged = cat([unm, dst])
```

The three reduction semantics of `scatter_reduce` (`include_self=True`, i.e. **the target's own old value participates in the reduction**):

| `reduce` | Result at target position `j` | Semantics |
|---|---|---|
| `"sum"` | `dst_old[j] + Σ(src[i] for i landing on j)` | sum |
| `"mean"` | `(dst_old[j] + Σ src[i]) / (1 + number of src landing on j)` | mean, **denominator includes the target itself** |
| `"amax"` | `max(dst_old[j], max(src[i] for i landing on j))` | maximum |

Returned fields:

- `merged`: 3-D list `[B][T - r][C]`, the merged token sequence (`r` is `len(src_idx[b])`);
- `merged_len`: the number of merged tokens;
- `dst_values`: 3-D list `[B][t2][C]`, the b-side values after `scatter_reduce`;
- `num_src`: `t1`;
- `num_dst`: `t2`.

Pitfalls:

- **The `mean` denominator is "1 + the number of incoming `src` tokens"**. If two `src` land on the same target whose old value is `10`, and the two `src` values are `1` and `2`, `mean` gives `(10 + 1 + 2) / 3 = 4.333`, **not** `1.5`.
- **Only `include_self=False` excludes the target** from the denominator. The official code uses the default `True`.
- **When `src_idx` is empty the official code returns `do_nothing`** (identity): `merged` is the input `x` itself, `merged_len = T`, and `dst_values` holds the untouched b-side values.
- **The `distill_token` interleaved layout**: the output is `[unm[0], dst[0], unm[1], dst[1], ...]`, not `[unm..., dst...]`. This keeps the distillation token adjacent to the class token in the sequence.
- **`merged` has length `T - r`**, not `T - 2r` (only `r` a-side tokens disappear; the b-side count is unchanged).

### 3. `size_weighted_merge`: size-weighted merging (`merge_wavg`)

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | 3-D list `[B][T][C]` | required |
| `size` | 3-D list `[B][T][1]`, how many original tokens each token stands for | required |
| `src_idx` / `dst_idx` / `unm_idx` | same as mode 2 | required |
| `distill_token` | whether to use the interleaved output layout | `False` |

Steps:

```
# 1. Weight by size first, then merge with the "sum" reduction
scaled = x * size                     # element-wise
x_merged = merge(scaled, reduce="sum")

# 2. The sizes themselves are reduced with the same merge rules
size_merged = merge(size, reduce="sum")

# 3. Divide by the reduced sizes
merged = x_merged / size_merged
```

Returned fields:

- `merged`: 3-D list `[B][T - r][C]`, the weighted-average tokens;
- `merged_size`: 3-D list `[B][T - r][1]`, the reduced sizes;
- `merged_len`: the number of merged tokens.

Pitfalls:

- **The denominator is the sum of sizes, not the token count**. A token whose size is `3` and which stays alone in the sequence has a denominator of `3` (not `1`).
- **`x` and `size` go through the same merge rules** (including the `distill_token` interleaved layout), so row `i` of `merged` and row `i` of `merged_size` **correspond strictly** — the division is row-aligned.
- **`size` has shape `[B][T][1]`** (channel length 1), not `[B][T]`.
- **When `src_idx` is empty the behaviour is `do_nothing`**: `merged` equals `x` and `merged_size` equals `size`.

### 4. `unmerge`: position refilling (restoring the original layout from the merged sequence)

Inputs:

| Parameter | Meaning | Default |
|---|---|---|
| `x` | the **merged** 3-D list `[B][T - r][C]` | required |
| `src_idx` / `dst_idx` / `unm_idx` | same as mode 2 | required |

Steps:

```
# 1. Split by the unm length: the front is the unmerged a side, the back is the b side
unm_len = len(unm_idx)
unm = x[:, :unm_len, :]
dst = x[:, unm_len:, :]

# 2. Each src's value is the value of the dst it merged into (post-merge value)
src = dst.gather(dst_idx)

# 3. Refill into the original layout
out = zeros(B, T, C)
out[:, 1::2, :] = dst            # b-side tokens go back to odd positions
out[2 * unm_idx] = unm           # unmerged a-side tokens go back to even positions
out[2 * src_idx] = src           # merged-away a-side tokens also go back to even positions (with merged values)
```

Returned fields:

- `out`: 3-D list `[B][T][C]`, the restored sequence;
- `out_len`: the original token count `T` (the token count of `x` plus `r`).

Pitfalls:

- **The input is the merged tensor** (length `T - r`); the output has `T` rows. Do not feed the pre-merge sequence in.
- **The merged-away `src` position is filled with the merged value** (the `dst` value), not with its own original value — `unmerge` restores the **layout**, not the values.
- **`2 * unm_idx` / `2 * src_idx` are even-position indices of the original sequence** (because the `a` side came from even positions).
- **The merged-away `src` and the unmerged `unm` are complementary on the even positions**, so the two rounds of scattering never overwrite each other.

## Scope of judging

- Only the four pure numerical ledgers above are judged: `match_indices` / `merge_reduce` / `size_weighted_merge` / `unmerge`.
- **Not judged**: the ViT forward pass, the actual `timm` patch wiring (`ToMeBlock` inserts the merge between attention and mlp), proportional attention (`attn + size.log()`), the multi-layer `r` scheduling of `parse_r`, the source matrix of `merge_source`, the `kth_bipartite_soft_matching` and `random_bipartite_soft_matching` variants, real speedups and accuracy losses on GPU, and real model weights.
- The `metric` / `x` / `size` in the problem statement are plain numeric lists containing **no real features**.
