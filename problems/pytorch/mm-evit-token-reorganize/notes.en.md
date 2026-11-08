# EViT Source Walkthrough: the Token Reorganization Ledger

> Official repository: `youweiliang/evit` (default branch **`master`**, not `main`)
> Files involved: `evit.py`, `helpers.py`

## 1. Where These Rules Live in EViT

EViT is a ViT variant with "token pruning" folded in: inside every Block, the
attention layer also decides "which patch tokens does this layer keep", and the
second half of the Block reorganizes the sequence accordingly.

Three official functions cover the three stages:

| Official location | Responsibility |
|---|---|
| `evit.py::Attention.forward` | compute the `cls_attn` scores, use `topk` to pick the tokens to keep |
| `evit.py::Block.forward` | gather the kept patches by `idx`, squash the discarded ones into `extra_token`, concatenate |
| `helpers.py::complement_idx` | the complement of `idx` in patch space |

The skeleton of `Attention.forward` (verbatim from upstream):

```python
left_tokens = N - 1
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
    left_tokens = math.ceil(keep_rate * (N - 1))
    if tokens is not None:
        left_tokens = tokens
    if left_tokens == N - 1:
        return x, None, None, None, left_tokens
    assert left_tokens >= 1
    cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
    cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
    # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
    # index = torch.cat([cls_idx, idx + 1], dim=1)
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

    return x, index, idx, cls_attn, left_tokens

return  x, None, None, None, left_tokens
```

The reorganization part of `Block.forward` (verbatim):

```python
tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
x = x + self.drop_path(tmp)

if index is not None:
    non_cls = x[:, 1:]
    x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

    if self.fuse_token:
        compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
        non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
        extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
    else:
        x = torch.cat([x[:, 0:1], x_others], dim=1)

x = x + self.drop_path(self.mlp(self.norm2(x)))
n_tokens = x.shape[1] - 1
```

---

## 2. `cls_attn`: The Scoring Semantics

### ★ Point 2.1: Only **row 0** of the attention is used

`attn` has shape `[B, H, N, N]` (query × key). The `0` in
`attn[:, :, 0, 1:]` sits on the **query** axis — the row where "CLS as query
asks every key". The other N-1 rows (patch as query) take **no part** in the
scoring.

### ★ Point 2.2: Columns start at **1** — the CLS column is dropped

`attn[b, h, 0, 0]` is CLS's attention weight on itself: always positive and
carrying no discriminative signal, so it is cut away. Hence `cls_attn` has
length **N-1**, not N.

### ★ Point 2.3: The cross-head reduction is an **arithmetic mean**

```python
cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
```

`dim=1` is the **head** axis (after `attn[:, :, 0, 1:]` the shape is
`[B, H, N-1]`). `mean` rather than `sum` because different heads' attention
scales are not strictly comparable — averaging lets each head cast one vote.

Measured (3 heads × 5 tokens):

```
cls_attn[0] = (0.1 + 0.3 + 0.1) / 3 = 0.16666...
```

---

## 3. `left_tokens`: Three Overlapping Rules

### ★ Point 3.1: The two variables in the trigger condition are **not the same thing**

That upstream line:

```python
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
```

- `self.keep_rate` is the **construction-time** value stored as
  `self.keep_rate` on the `Attention` module;
- `keep_rate` is the **call-time** argument (a parameter of `forward`, which
  defaults to `self.keep_rate`).

Since `and` binds tighter than `or` in Python, the whole line means:

```python
(self.keep_rate < 1 and keep_rate < 1) or (tokens is not None)
```

**Both ratios must be `< 1`** for the keep-rate branch. `tokens`, however, can
trigger it **on its own**.

Measured:

| `self.keep_rate` | call-time `keep_rate` | `tokens` | triggered? |
|---|---|---|---|
| 0.5 | 0.5 | None | yes |
| **1.0** | 0.5 | None | **no** (`self.keep_rate < 1` is False) |
| 0.5 | 1.0 | None | no (`keep_rate < 1` is False) |
| **1.0** | 0.5 | **3** | **yes** (`tokens is not None` alone suffices) |

This is what the upstream comment `# double check the keep rate` is warning
about — a module constructed with `keep_rate=1.0` will **not** enter the
pruning branch even when the caller passes `keep_rate=0.5`.

### ★ Point 3.2: `left_tokens = math.ceil(keep_rate * (N - 1))`

- **Ceiling** (`ceil`), not floor, not round;
- the base is **`N - 1`** (N includes the CLS token, i.e. the patch count).

Measured: `N = 197` (196 patches), `keep_rate = 0.7` → `ceil(0.7 * 196) = 138`.

### ★ Point 3.3: `tokens` **overwrites** directly

```python
if tokens is not None:
    left_tokens = tokens
```

A **direct assignment** — not `min`, not `max`. A `tokens` larger than the ceil
result is used as-is.

### ★ Point 3.4: **Early return** when `left_tokens == N - 1`

```python
if left_tokens == N - 1:
    return x, None, None, None, left_tokens
```

The condition **was** triggered (`triggered` stays `True`), but since "keep
everything" needs no selection, `index` / `idx` are `None`.

Measured: `N = 10`, `keep_rate = 0.9` → `ceil(0.9 * 9) = 9 == N - 1` → early
return; `keep_rate = 0.888` → `ceil(7.99) = 8` → proceeds normally.

### ★ Point 3.5: `assert left_tokens >= 1`

Keeping fewer than one token is illegal and the official code asserts.

---

## 4. `topk` and `complement_idx`

### ★ Point 4.1: `topk`'s output is in **descending score** order

```python
_, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
```

`sorted=True` means the returned values are ordered **by descending score**,
**not** by ascending original index.

Measured: `cls_attn = [0.1, 0.5, 0.3, 0.05, 0.05]`, `left_tokens = 3`
→ `idx = [1, 2, 0]` (scores 0.5, 0.3, 0.1), whereas ascending index order
would be `[0, 1, 2]`.

This order propagates all the way into the gather in `Block.forward` — in other
words, **the retained patches are laid out in the sequence by descending
score**; positional information is reshuffled.

### ★ Point 4.2: Ties break by ascending index

`torch.topk` takes tied scores in **ascending index** order. Measured:

- `[0.5, 0.5, 0.5, 0.1]` topk(2) → `[0, 1]`
- `[0.1, 0.5, 0.5, 0.5]` topk(2) → `[1, 2]`
- `[0.3, 0.3, 0.3, 0.3]` topk(2) → `[0, 1]`

### ★ Point 4.3: `complement_idx` returns the complement in **ascending index** order

The upstream implementation (`helpers.py`):

```python
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
```

The key is `torch.sort(..., descending=False)` — the result is **ascending**.

Measured: `complement_idx([[1, 3]], dim=4)` → `[[0, 2]]`;
`complement_idx([[0, 1, 2]], dim=4)` → `[[3]]`;
`complement_idx([[0, 1, 2, 3]], dim=4)` → shape `(1, 0)`.

---

## 5. `extra_token`: A Weighted Sum, Not a Weighted Average

### ★ Point 5.1: `torch.sum` — a sum

```python
extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
```

`torch.sum` is a **sum**. Many implementations divide by the sum of the weights
to turn it into an average — the official code does not.

Measured: `non_cls = [[1], [2], [3], [4]]`, `cls_attn = [0.1, 0.2, 0.3, 0.4]`,
discarding patches 2 and 3 → `extra_token = 0.3 * 3 + 0.4 * 4 = 2.5`
(the weighted average would be `2.5 / 0.7 = 3.5714...`).

### ★ Point 5.2: The weights are **not normalized**

The weights are the raw `cls_attn` values (small decimals after softmax), with
no normalization whatsoever. So the magnitude of `extra_token` scales with "the
total attention of the discarded tokens".

### ★ Point 5.3: `non_topk` and `non_topk_attn` share one order

Both are gathered with `compl`, so **both** follow the ascending complement
order. If one followed the complement order and the other something else, the
weights would be misaligned.

### ★ Point 5.4: `extra_token` goes **last**

```python
x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
```

The sequence is `[CLS] + [kept patches] + [extra_token]` — `extra_token` is at
the **end**, not at the front.

---

## 6. The Execution Order in `Block.forward`

### ★ Point 6.1: Reorganization happens **after** the residual

```python
tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
x = x + self.drop_path(tmp)          # ← attention residual first

if index is not None:
    non_cls = x[:, 1:]               # ← patches taken from the【post-residual】x
    ...
x = x + self.drop_path(self.mlp(self.norm2(x)))   # ← MLP last
```

In other words, the gathered patches are the vectors **after** the attention
residual was added, not before. Hand-reproducing `Block.forward` against the
official one gives max abs diff = 0.

### ★ Point 6.2: The MLP runs **after** the reorganization

`self.mlp(self.norm2(x))` acts on the **shortened** sequence. Once fewer tokens
survive, the MLP's compute shrinks too — that is exactly where the pruning
saves work.

### ★ Point 6.3: `n_tokens` is the output length minus 1

```python
n_tokens = x.shape[1] - 1
```

The CLS is subtracted. With fuse, `n_tokens = left_tokens + 1` (the extra one is
`extra_token`); without fuse, `n_tokens = left_tokens`.

---

## 7. Three Pitfalls When Reproducing the Official Code

### Pitfall 1: `topk`'s descending order is not "incidental"

The intuition for "select the top-k" is to write `sorted(range(n), key=lambda
i: -score[i])` and take the first k — that is **ascending** order. What the
official code wants is `torch.topk`'s **descending** output. The two agree on
the *set* but differ in *order*, and the order directly determines how the
reorganized sequence is laid out.

Verification trick: use scores `[0.1, 0.5, 0.3, 0.05, 0.05]`; the official
gives `idx = [1, 2, 0]`.

### Pitfall 2: The dual identity of `self.keep_rate` and `keep_rate`

Almost everyone reads that line as a single boolean expression the first time.
But `self.keep_rate` (construction-time) and `keep_rate` (call-time) are
**two variables**. A module constructed with `keep_rate=1.0` will **not**
trigger pruning even if the caller passes `keep_rate=0.5`.

Verification trick: use the official `Attention` class, construct it with
`keep_rate=1.0`, then call with `keep_rate=0.5` and `tokens=None` →
`index is None`.

### Pitfall 3: The magnitude of `extra_token`

Because the weights are not normalized, the absolute value of `extra_token`
floats with "the total attention of the discarded tokens". If your output has
the stable magnitude of a "mean", you almost certainly divided once too often.

Verification trick: with `cls_attn` all zeros, `extra_token` must be **all
zeros** (not NaN); with a weight sum of `0.7`, `weight_sum` must be `0.7`, not
`1.0`.

---

## 8. The Three-Layer Verification Structure

The verification here has three layers, and no single one is sufficient:

1. **Official layer**: slice `Attention` / `Block` / `Mlp` / `DropPath` /
   `complement_idx` verbatim out of upstream `evit.py` / `helpers.py` with AST,
   stripping only the `timm` dependency (the evaluation environment has no
   `timm`), and run the official code itself.
2. **Reference layer**: the pure Python + `Fraction` implementation (this
   problem's `solution.py`).
3. **oracle layer**: a pure-float reimplementation written independently from
   the official formulas, with no dependency on the reference layer.

Measured across 34 cases, all three layers agree to within `2.776e-17`.

**Why three layers**: the official code has several places where it "runs fine
but means something else" — the `topk` order, the dual identity of
`self.keep_rate`. If you only test against your own reference, an
implementation that writes `topk` in ascending order may "accidentally" pass on
some cases. Pinning the semantics requires the **official layer** as the
baseline.

---

## 9. Three Principles for Mutation Testing

This problem's mutation list has 37 entries and ends at **37/37 CAUGHT**. The
mistakes made along the way:

1. **Equivalent mutants must be removed proactively**. Two survived the first
   round; checking each one showed they were **truly equivalent**:
   - turning `per_head.append(hr[0][1:])` into "flatten all rows then take
     1..N-1" — after flattening, the first N-1 elements are still row 0's
     `1:`, so the **numbers are exactly the same**;
   - turning `out = [cls_row] + kept` into `out = ([cls_row] + kept)` — pure
     parentheses.

   A survivor here does not mean weak judging; first check whether the mutant
   is equivalent, then either delete it or replace it with an equivalent
   rewrite that **really does produce different numbers** (e.g. "sum over all
   rows").

2. **The validation layer must be split into mutually independent defenses**.
   Every check here simultaneously asserts:
   - "the result equals the oracle";
   - "the result **differs from** that plausible-but-wrong candidate".

   For `extra_token`, for instance, we assert equality with `2.5` *and*
   inequality with `2.5 / 0.7` (the weighted average) and `7.0` (the
   unweighted sum). With only the former, an implementation that "happens to
   agree on some case whose weight sum is 1" would slip through.

3. **Unobservable differences need a purpose-built case, and the anchor must
   land on an input that actually changes**. Three survivors in the first round
   were all **case-coverage gaps**:
   - `tokens` overwrite written as `min(tokens, ceil)`: only observable when
     `tokens > ceil`, so a case with `keep_rate=0.3` (ceil = 2) and
     `tokens=4` was added;
   - the `left_tokens >= 1` assert relaxed to `>= 0`: only observable when
     `tokens=0`, so a case with `tokens=0` expecting an exception was added;
   - `kept_positions` missing the `+ 1`: only asserted on the **fused** branch,
     so the non-fused branch got a `kept_positions` assertion too.

   The anchors must also land where the numbers really change — three mutants
   in the first round reported `ANCHOR-ERROR` because the anchor indentation was
   wrong (16 spaces vs. the actual 12), so the real indentation has to be
   checked one by one.

---

## 10. Index of the Official Source Locations

| Symbol | Location |
|---|---|
| `Attention.forward` selection branch | `evit.py:207-224` |
| `cls_attn = attn[:, :, 0, 1:]` | `evit.py:215` |
| `cls_attn = cls_attn.mean(dim=1)` | `evit.py:216` |
| `torch.topk(..., largest=True, sorted=True)` | `evit.py:217` |
| `index = idx.unsqueeze(-1).expand(-1, -1, C)` | `evit.py:220` |
| `Block.forward` reorganization block | `evit.py:250-266` |
| `x = x + self.drop_path(tmp)` | `evit.py:251` |
| `non_cls = x[:, 1:]` | `evit.py:255` |
| `x_others = torch.gather(non_cls, dim=1, index=index)` | `evit.py:256` |
| `compl = complement_idx(idx, N - 1)` | `evit.py:259` |
| `extra_token = torch.sum(...)` | `evit.py:263` |
| `torch.cat([x[:, 0:1], x_others, extra_token], dim=1)` | `evit.py:264` |
| `n_tokens = x.shape[1] - 1` | `evit.py:269` |
| `complement_idx` | `helpers.py` |
