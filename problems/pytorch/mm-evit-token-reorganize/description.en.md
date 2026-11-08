# EViT Token Reorganization Ledger

## Background

There is a simple way to make a vision Transformer cheaper: **do not carry every
patch token into the next layer**. But "which ones do we drop" cannot be decided
arbitrarily — drop the wrong ones and the model simply stops seeing those regions.

EViT (*Early Vision Transformer*) offers a clean answer: **let the CLS token vote**.

Its core observation is that the attention row computed with the CLS token as the
query is, by construction, a readout of "where globally matters". So EViT does
three things:

1. **Score**: take row 0 of the attention (CLS as query), drop the CLS column
   itself, and average across all heads to get one importance score per patch.
2. **Select**: keep the top-k patches by that score.
3. **Reorganize**: re-order the surviving patches, then fold **all** discarded
   patches into **one** `extra_token` by **weighted sum** and append it at the end
   of the sequence.

Step 3 is the interesting design: discarded tokens are not simply thrown away —
their information is "squashed" into a single token that travels downstream, so
later layers can still sense "something used to be over there".

The official source lives in `youweiliang/evit` (default branch `master`):

| Location | Role |
|---|---|
| `evit.py::Attention.forward` | cls_attn scoring + topk selection |
| `evit.py::Block.forward` | token reorganization (gather / fuse) |
| `helpers.py::complement_idx` | complement computation |

Your job is to reproduce the exact numerical semantics of these three functions
in pure Python. There are at least eight places where the official code does
something other than what "looks reasonable", and the judge pins every one of
them down.

## Conventions

Every input is a **plain Python list**; no tensor shape inference is involved:

- `attn`: **`[H][N][N]`** — H heads, each an N×N attention matrix. N includes the
  CLS token, i.e. `N = patch count + 1`. Every row of every head should be a
  non-negative attention distribution.
- `hidden`: **`[N][C]`** — N tokens, C channels, **row 0 is the CLS token**.
- `non_cls`: **`[N-1][C]`** — the patch sequence after CLS has been removed;
  index `i` corresponds to token `i+1` of the original sequence.
- `cls_attn`: a scoring row of length `N-1`; index `i` corresponds to patch `i`.
- `idx`: the indices of the retained patches in **patch space** (0-based, CLS
  excluded).
- `complement`: the complement of `idx` in patch space.
- All outputs are plain `float` (anything that can be an exact rational goes
  through `fractions.Fraction` internally).
- Illegal inputs raise `ValueError`.

---

## mode 1: `cls_attn_score`

Reproduce the official scoring semantics:

```python
cls_attn = attn[:, :, 0, 1:]   # [B, H, N-1]
cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
```

**Signature**

```python
cls_attn_score(attn)
    → {"cls_attn": [...], "n_heads": int, "n_tokens": int}
```

**Key points**

1. ★ Only **row 0** of the attention is used — the row where **CLS is the query**.
   All other rows (patch as query) take **no part** in the scoring.
2. ★ The columns start at **1** — the **CLS column itself is dropped** (CLS's
   attention weight on itself carries no information). Hence the output length is
   `N-1`, not `N`.
3. ★ The cross-head reduction is an **arithmetic mean** (`mean`), not a sum and
   not a max.
4. `n_heads` = H, `n_tokens` = N.

**Example**

```python
attn = [
    [[0.5, 0.1, 0.2, 0.1, 0.1],   # head 0, the CLS row
     [0.1, 0.4, 0.2, 0.2, 0.1],
     [0.1, 0.1, 0.4, 0.2, 0.2],
     [0.1, 0.1, 0.2, 0.4, 0.2],
     [0.1, 0.2, 0.1, 0.2, 0.4]],
    [[0.2, 0.3, 0.2, 0.2, 0.1],   # head 1, the CLS row
     [0.2, 0.2, 0.3, 0.2, 0.1],
     [0.1, 0.2, 0.2, 0.3, 0.2],
     [0.2, 0.1, 0.2, 0.2, 0.3],
     [0.3, 0.1, 0.2, 0.1, 0.3]],
]
solve("cls_attn_score", attn=attn)["cls_attn"]
# → [(0.2 + 0.3) / 2, (0.2 + 0.2) / 2, (0.1 + 0.2) / 2, (0.1 + 0.1) / 2]
#   = [0.25, 0.2, 0.15, 0.1]
```

---

## mode 2: `select_tokens`

Reproduce the official token-count decision and top-k selection:

```python
left_tokens = N - 1
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
    left_tokens = math.ceil(keep_rate * (N - 1))
    if tokens is not None:
        left_tokens = tokens
    if left_tokens == N - 1:
        return x, None, None, None, left_tokens
    assert left_tokens >= 1
    cls_attn = attn[:, :, 0, 1:]
    cls_attn = cls_attn.mean(dim=1)
    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
    index = idx.unsqueeze(-1).expand(-1, -1, C)
    return x, index, idx, cls_attn, left_tokens
return x, None, None, None, left_tokens
```

**Signature**

```python
select_tokens(cls_attn, keep_rate, self_keep_rate=1.0, tokens=None)
    → {"left_tokens": int, "triggered": bool, "idx": [...] | None,
       "scores": [...] | None, "kept_scores": [...] | None,
       "complement": [...] | None, "n_tokens": int}
```

**Key points**

1. ★ The two variables in the trigger condition are **different things**:
   - `self_keep_rate` is the **construction-time** value stored as
     `self.keep_rate` on the `Attention` module;
   - `keep_rate` is the **call-time** argument.

   Since `and` binds tighter than `or` in Python, the official line
   `self.keep_rate < 1 and keep_rate < 1 or tokens is not None` is equivalent to
   `(self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)`.
   Therefore: **both ratios must be `< 1`** for the keep-rate branch, while
   `tokens` can trigger it **on its own**.
2. ★ `left_tokens = math.ceil(keep_rate * (N - 1))` — **ceiling**, and the base
   is **`N - 1`** (N includes the CLS token), not N.
3. ★ When `tokens` is not `None`, `left_tokens` is **overwritten** with `tokens`
   (the ceil result is discarded). The overwrite is a **direct assignment**, not
   a `min` — a `tokens` larger than the ceil result is used as-is.
4. ★ When `left_tokens == N - 1` the function **returns early**: no token is
   selected and `idx` / `scores` / `kept_scores` / `complement` are all
   **`None`**. Note the condition *was* triggered (`triggered` stays `True`) —
   it just short-circuits because "keeping everything" needs no selection.
5. `left_tokens` must be `>= 1`, otherwise the official `assert` fails — your
   implementation must raise.
6. ★ `torch.topk(..., largest=True, sorted=True)` returns its values **ordered
   by descending score**, **not** by ascending original index. Tied scores are
   broken by ascending index.
7. `complement` is the complement of `idx` in patch space (length `N-1`), in
   **ascending index** order (the official `complement_idx` does an internal
   `torch.sort(descending=False)`).
8. `scores` and `kept_scores` are the scores of the retained tokens, in `idx`
   order; they are `None` on the early-return path.
9. `n_tokens` is always `N` (= patch count + 1).

---

## mode 3: `fuse_token`

Reproduce the official step that folds the discarded patches into one
`extra_token`:

```python
compl = complement_idx(idx, N - 1)                              # [B, N-1-left_tokens]
non_topk = torch.gather(non_cls, dim=1,
                         index=compl.unsqueeze(-1).expand(-1, -1, C))
non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1),
                        dim=1, keepdim=True)                    # [B, 1, C]
```

**Signature**

```python
fuse_token(non_cls, cls_attn, idx, complement)
    → {"extra_token": [...], "non_topk": [[...]], "non_topk_attn": [...],
       "weight_sum": float}
```

**Key points**

1. ★ `extra_token` is a **weighted sum**, **not** a weighted average:
   `sum_i(w_i * x_i)`. The most common mistake is dividing by the sum of the
   weights to turn it into an average.
2. ★ The weights are the raw `cls_attn` values (small decimals after softmax),
   used **without any normalization**. So the magnitude of `extra_token` scales
   with the weight sum — it is not a "mean".
3. `non_topk` and `non_topk_attn` are both ordered by `complement`
   (`complement` itself is in ascending index order).
4. `weight_sum` is the sum of the discarded tokens' weights (**not normalized**);
   it serves as evidence that a weighted sum — not an average — was performed.
5. `idx` itself takes no part in the numeric computation (the complement is
   supplied via `complement`), but it must still be consistent with the lengths
   of `non_cls` / `cls_attn`.

**Example**

```python
non_cls = [[1.0], [2.0], [3.0], [4.0]]
cls_attn = [0.1, 0.2, 0.3, 0.4]
complement = [2, 3]     # discard patches 2 and 3
solve("fuse_token", non_cls=non_cls, cls_attn=cls_attn,
      idx=[0, 1], complement=complement)["extra_token"]
# → [0.3 * 3.0 + 0.4 * 4.0] = [2.5]
#   (not (0.3*3 + 0.4*4) / (0.3+0.4) = 3.5714...)
```

---

## mode 4: `reorganize`

Reproduce the reorganization in the official `Block.forward`:

```python
non_cls = x[:, 1:]
x_others = torch.gather(non_cls, dim=1, index=index)   # [B, left_tokens, C]

if self.fuse_token:
    compl = complement_idx(idx, N - 1)
    non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
else:
    x = torch.cat([x[:, 0:1], x_others], dim=1)
```

**Signature**

```python
reorganize(hidden, idx, extra_token=None, fuse=True)
    → {"out": [[...]], "n_tokens": int, "kept_positions": [...],
       "extra_token": [...] | None}
```

**Key points**

1. ★ The output sequence is **`[CLS] + [patches gathered in idx order] +
   ([extra_token] if fuse)`**.
2. ★ `extra_token` goes **last**, not first. Inserting it right after CLS is the
   most common mistake.
3. ★ The gathered patches keep the order of `idx` (**descending score**),
   **not** ascending original index.
4. `hidden` is the sequence **after** the attention residual has been added —
   the official code reorganizes only after `x = x + drop_path(tmp)`.
5. Output length: with fuse, `1 + len(idx) + 1`; without fuse,
   `1 + len(idx)`.
6. `n_tokens` = output length - 1 (drop the CLS).
7. `kept_positions` are the positions of the retained patches in the
   **original** sequence (patch index `i` → original row `i+1`, since row 0 is
   the CLS).
8. Without fuse, the `extra_token` output field is `None`.

---

## Common Mistakes

1. Averaging `cls_attn` over **all rows** instead of taking only row 0 (the row
   where CLS is the query).
2. Including the **CLS column itself** in `cls_attn` (making the output length
   N).
3. Summing or taking the **max** across heads instead of the arithmetic mean.
4. Writing the trigger condition as `keep_rate < 1` — ignoring the
   **construction-time** `self.keep_rate`.
5. Writing the trigger condition as `self_keep_rate < 1 and keep_rate < 1` —
   dropping the path where `tokens` triggers it **on its own**.
6. Using **floor** or **round** for `left_tokens` instead of `ceil`.
7. Using **N** as the base of `left_tokens` instead of **N - 1**.
8. Writing the `tokens` overwrite as `min(tokens, left_tokens)` instead of a
   **direct assignment**.
9. Not taking the **early return** when `left_tokens == N - 1` (going on to
   select a top-k anyway).
10. Returning an empty list for `idx` on the early-return path instead of
    `None`.
11. Returning the top-k result in **ascending index** order instead of
    descending score.
12. Returning `complement` in **descending** index order (the official one is
    ascending).
13. Computing `extra_token` as a **weighted average** (dividing by the weight
    sum).
14. **Normalizing** the weights of `extra_token`.
15. Ignoring the weights of `extra_token` (plain sum over patches).
16. Placing `extra_token` at the **front** of the sequence (right after CLS)
    instead of at the end.
17. Letting the **CLS** into the patch sequence in `reorganize` (`hidden[1:]`
    written as `hidden`).
18. Dropping the **CLS** row in `reorganize`.
19. Computing `n_tokens` as the output length (forgetting to subtract 1 for the
    CLS).
20. Returning patch-space indices for `kept_positions` instead of original
    sequence positions.

## Self-check

```bash
python starter.py            # skeleton self-check
```

Before submitting, confirm:

- The return key names of the four modes exactly match the statement.
- All outputs are `float` (not `Fraction`).
- Illegal inputs raise `ValueError`.
- On the early-return path of `select_tokens`, `idx` / `scores` /
  `kept_scores` / `complement` are all `None`.
