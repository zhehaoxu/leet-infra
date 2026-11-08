# Official mechanism and pitfall notes

## Official source and version

- Repository: `facebookresearch/ToMe`, default branch `main` (an actively maintained repo with ~1205 stars according to the GitHub API).
- Files used:
  - `tome/merge.py` (240 lines): `do_nothing` / `bipartite_soft_matching` / `kth_bipartite_soft_matching` / `random_bipartite_soft_matching` / `merge_wavg` / `merge_source`;
  - `tome/utils.py`: `parse_r`;
  - `tome/patch/timm.py` (150 lines): `ToMeBlock.forward` / `ToMeAttention.forward` / `apply_patch`.
- Retrieval: GitHub API `git/trees/main?recursive=1` + `contents?ref=main` + `base64.b64decode`.
- The official cross-check script uses `.venv-eval/bin/python` (torch 2.14.1, no GPU needed — the official implementation is pure tensor ops).

## Line-by-line anatomy of the official `bipartite_soft_matching`

```python
protected = 0
if class_token:  protected += 1
if distill_token: protected += 1
t = metric.shape[1]
r = min(r, (t - protected) // 2)
if r <= 0:
    return do_nothing, do_nothing
with torch.no_grad():
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    if class_token:  scores[..., 0, :] = -math.inf
    if distill_token: scores[..., :, 0] = -math.inf
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[..., r:, :]
    src_idx = edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    if class_token:
        unm_idx = unm_idx.sort(dim=1)[0]
```

Mechanisms pinned one by one:

1. **`-inf` protection lives on two different axes**: `class_token` → `scores[..., 0, :]` (**row**); `distill_token` → `scores[..., :, 0]` (**column**). This is the main judging point of the problem; writing both on the same axis is entirely wrong.
2. **The cap on `r` is `(t - protected) // 2`**, where `protected` counts `class_token` and `distill_token` **one each**. With both enabled the cap is `(t - 2) // 2`.
3. **When `r <= 0` the code returns `do_nothing`** (`lambda x, mode=None: x`), i.e. **complete identity**: no matching, no merging, length unchanged.
4. **Alternating split**: `a = metric[..., ::2, :]` (even positions), `b = metric[..., 1::2, :]` (odd positions). When `t` is odd, `a` is one longer than `b`.
5. **`unm_idx` is sorted only under `class_token`** (`unm_idx.sort(dim=1)[0]`, ascending along the token dim), so that the class token comes first. With `distill_token` alone it is **not** sorted.
6. **`src_idx` and `unm_idx` are the two complementary pieces of `edge_idx`**: `src_idx` is the first `r` (by descending similarity), `unm_idx` is the rest.
7. **`dst_idx` may contain duplicates**: several `src` tokens may match the same `b`.

## Anatomy of the official `merge` closure

```python
def merge(x, mode="mean"):
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape
    unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=src_t.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
    if distill_token:
        return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
    return torch.cat([unm, dst], dim=1)
```

- **`scatter_reduce`'s `include_self` defaults to `True`**. Measured (`dst=[10,20,30]`, `src=[1,2,4]`, `idx=[0,0,1]`):
  - `sum` → `[13, 24, 30]`;
  - `mean` → `[4.333, 12, 30]` (i.e. `(10+1+2)/3`, **denominator includes the target itself**);
  - `amax` → `[10, 20, 30]` (also includes the target).
  - Only `include_self=False` gives `mean` = `[1.5, 4.0, 30.0]`.
  - Measured: duplicate indices are **deterministic** on CPU (identical results across runs).
- **The `distill_token` interleaved layout**: `cat([unm[:1], dst[:1], unm[1:], dst[1:]])`, which keeps the distillation token adjacent to the class token.
- **The output length is `t - r`** (only `r` a-side tokens disappear).

## Anatomy of the official `merge_wavg`

```python
def merge_wavg(merge, x, size=None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size
```

- **The denominator is the sum of sizes, not the token count**. A token whose size is `3` and which stays alone in the sequence has a denominator of `3`.
- **`x` and `size` go through the same `merge` closure**, so the `distill_token` interleaved layout applies to both — `merged` and `merged_size` correspond row by row.
- **`size` has shape `[..., 0, None]`**, i.e. `[B, T, 1]`.

## Anatomy of the official `unmerge` closure

```python
def unmerge(x):
    unm_len = unm_idx.shape[1]
    unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
    src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
    out = torch.zeros(n, metric.shape[1], c, ...)
    out[..., 1::2, :] = dst
    out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
    out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
    return out
```

- **The input is the merged tensor** (length `t - r`); the output has `t` rows. `out`'s length comes from `metric.shape[1]` (the original token count).
- **The value of `src` is the post-merge `dst` value**, not its own original value — `unmerge` restores the **layout**, not the values.
- **`2 * unm_idx` / `2 * src_idx`** are even-position indices of the original sequence.

## Floating-point tails and ties: where this problem's convention comes from

The official code is float64 `metric / metric.norm()` + `a @ b.T`. Three empirical findings:

1. **torch `max(dim=-1)` takes the smallest index on ties** (`[1,1,1,-1]` → index 0);
2. **torch `argsort(dim=-1, descending=True)` is order-preserving and stable on CPU** (`[0.5,0.5,0.5,0.9,0.5]` → `[3,0,1,2,4]`);
3. **Normalization introduces tails**: normalizing `[1,1]` and `[2,2]` gives a dot product of `0.9999999999999998` rather than `1.0`.

Finding 3 is fatal: in cases where the **exact cosines are equal**, the official floating point breaks the tie because of a 1e-17 tail, so the item chosen by `max` and the order produced by `argsort` **depend on the rounding bits** — the same input may yield a different merge plan on a different BLAS or machine.

This problem therefore pins the convention to an **exact rational** one: `metric` / `x` / `size` are all integers, comparisons use integer cross-multiplication to eliminate square roots, and the tie rules are written out explicitly (`max` takes the smallest index; the sort is stable and order-preserving).

**The connected requirement**: the official floating point and the exact convention agree on all **non-degenerate** cases and diverge only where the cosines are **exactly tied**. The official cross-check must be designed around that fact — one cannot use official floating point as a pointwise oracle. Instead it must **locate each divergence and prove its exact cosine difference is <= 1e-12** (mathematically equal, flipped by floating-point noise). See below.

## Verification approach and test design

- **Official cross-check (three layers)**:
  1. **Primary oracle**: a pure-Python implementation using `fractions.Fraction` for exact rational comparison — an **independent code path** from the reference solution's integer cross-multiplication (Fraction compares exact quotients directly);
  2. **Official cross-validation**: the official float64 implementation from `tome/merge.py`. Of 600 `match_indices` cases, **567 have identical indices**; the other **33 divergences** were all localized to "exact cosines nearly equal (<= 1e-12, floating-point tail)" with no exceptions — i.e. the official and exact conventions differ **only where the math is genuinely tied**;
  3. **Merge mechanism cross-validation**: run our exactly-computed indices through the official `gather` / `scatter_reduce` / `cat` / `scatter_` semantics to verify the `include_self` behaviour of gather/scatter_reduce, the output reordering, the `distill_token` interleaved layout, the `merge_wavg` denominator, and the `unmerge` refill.
  - Final tally: `match_indices` 600 + `merge_reduce` 510 + `size_weighted_merge` 504 + `unmerge` 520 cases, **all consistent**.
- **Test cases**: 4 `check_*` functions (`check_match_indices` / `check_merge_reduce` / `check_size_weighted_merge` / `check_unmerge`), each with an independent pure-Python oracle (Fraction exact cosines), hand-computed anchors, and grid cross-checks.
- **Hand-computed anchors pin down**: `max` ties taking the smallest index, `class_token` protecting the row vs `distill_token` protecting the column, `unm` being sorted only under `class_token`, `r` being capped by `(T - protected) // 2`, the `do_nothing` special case for `r <= 0`, negative cosines still being selectable, the `mean` denominator when several `src` land on one `dst`, the interleaved output layout under `distill_token`, `merge_wavg`'s denominator being the sum of sizes, `unmerge` refilling post-merge values, and `unmerge`'s output being `r` longer than its input.
- **Mutation testing**: 47 single-point mutations, all CAUGHT, with a **correct-solution self-check passing first**.

## Pitfalls hit during cross-checking and mutation (reusable lessons)

1. **Official floating point cannot serve as a pointwise oracle.** The first cross-check version used official float64 as the expected value and blew up at `match[382]` (`dst_idx[1][3]`: official `3`, self-written `1`). The diagnosis showed the root cause was **a floating-point tail flipping the `max` choice where the exact cosines were equal** — not an implementation bug. **Cross-checks must be designed as "divergence localization + exact difference <= tail magnitude" against official float, not as pointwise equality.**
2. **The divergence localization must index correctly.** The first localization function compared `nmax` at `order[pos-1]` and `order[pos]` (indexing the exact `nmax` with indices from the **exact** order), which cannot possibly line up with the offending pair. The correct approach is to compare, **position by position**, the exact `nmax` of the two elements the official order and the exact order place at that position.
3. **Do not invert the assertion direction.** `locate_official_divergence` returning `None` means "all divergences are within the tail magnitude" (a pass); returning a dict means "beyond the magnitude — a real bug". The first version had them swapped, so a passing case still reported "could not locate the divergence".
4. **The `merge` closure called inside `merge_wavg` handles `distill_token`.** The first version merged `size` as a plain `cat([unm, dst])`, which does not match the official code; officially it is the **same closure** applied to both `x` and `size`, so both use the interleaved layout.
5. **The input of `unmerge` is the merged tensor.** The first cross-check passed the **original** `x` to the official `unmerge`, which expects length `t - r`; meanwhile the self-written implementation treated "post-merge length" as "original length". Both were wrong. The correct relation is `t = m + r`.
6. **The `scaled` tensor of `_size_weighted_merge` was missing its batch dimension.** The first version built `scaled` **inside** the batch loop, so `_merge_reduce` received a single-batch slice and raised `TypeError: 'int' object is not iterable`. The whole batch's `scaled` must be built before it is passed.
7. **`src` must be pairwise distinct.** The official `src_idx` is the first `r` of an `argsort`, so it is **necessarily distinct**; it is `dst_idx` that may repeat (several `a` tokens pointing at the same `b`). The first grid let `src` repeat (`rng.choice` with replacement), which broke `unm`'s length relative to `t1 - r` and the length arithmetic in `unmerge`. The grid must pick `src` **without replacement** and let only `dst` repeat randomly.
8. **Recheck hand-computed anchors' divisions yourself.** The `merge_wavg` anchors were wrong three times in a row (`18/3` written as `12/3`, `50/5` as `30/5`, and `merged_size` written as `[[3],[3],[4]]` instead of `[[3],[2],[5]]`). The self-check caught all of them immediately — **when the anchor is wrong and the solution is right, fix the anchor, not the solution**.
9. **True-equivalent mutations must be removed.** `dot-sum-order` (reversing the summation order of the dot product) survived because **integer addition is associative** and therefore order-independent. It was replaced with the measurable form `dot-drop-last` (dropping the last term).
10. **Mutation anchors must match the current code verbatim.** The anchors of `wavg-size-not-merged` and `cos-value-none-nan` reported `ANCHOR-ERROR` because of indentation/comment mismatches; after fixing the anchor text they behaved normally.

## Out-of-scope boundaries

The ViT forward pass and the actual `timm` patch wiring (`ToMeBlock` inserts the merge between attention and mlp), proportional attention (`attn + size.log()`), the multi-layer `r` scheduling of `parse_r`, the source matrix of `merge_source`, the `kth_bipartite_soft_matching` and `random_bipartite_soft_matching` variants, real speedups and accuracy losses on GPU, and real model weights are all out of scope. This problem keeps only four deterministic pure-numerical ledgers.
