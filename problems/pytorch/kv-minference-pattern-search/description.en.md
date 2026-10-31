# KV Cache: MInference Sparse Attention Pattern Search

## Problem

Full attention during long-context prefill costs \(O(N^2)\), but real models attend highly sparsely and in **predictable shapes**: a handful of "vertical" columns everybody looks at (the attention sink plus a few key tokens), plus a "slash" band hugging the diagonal. MInference (2407.02490) `search_pattern` first runs a cheap **pattern search**: over a batch of `(kind, size)` candidates it estimates "how much attention mass does this mask keep", each head takes the highest-scoring one, and only then does the real sparse attention run with the chosen pattern.

The scoring needs no kernel at all. Let \(P\) be the true causal attention (one \(N\times N\) row-stochastic lower triangle per head) and \(M\) the 0/1 candidate mask; the score is

\[
\text{score}=\frac{1}{N}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} P_{ij}\,M_{ij}
\]

i.e. the mass kept per row averaged over **all** \(N\) rows, and it lives in \([0,1]\). The three `kind`s build \(M\) as follows:

- `stream_llm(a, b)`: first \(a\) columns \(\cup\) the last \(b+1\) diagonals (main diagonal included), intersected with the causal lower triangle;
- `vertical_and_slash(a, b)`: looks only at the **last 64 queries** of the attention, computing per-column mass `vertical` and per-diagonal mass `slash`; the first 30 columns and the last 30 diagonals are forced to stay by being set to \(+\infty\), and the rest are taken as the top-\(a\) columns and top-\(b\) diagonals, then unioned;
- `block_sparse(a)`: zero-pad \(q\) and \(k\) to \(\lceil N/32\rceil\cdot 32\) rows and average every 32 rows into a block vector (**the padded zeros stay in the denominator**); block-level dot products **without** \(\sqrt D\), plus a block-causal mask; per row the lowest-scoring \(\texttt{block\_num}-\lfloor \texttt{block\_num}/\texttt{a}\rfloor\) blocks are dropped, and the block grid is expanded back to a token grid and intersected with the causal mask.

Three details that are easy to get wrong:

1. \(P\) is computed once and shared by all candidates; the score is **not renormalized**, so a row whose mask is entirely empty contributes 0;
2. the chosen `slash` diagonals are **unioned** onto the vertical result, independently of whether that column lost the vertical contest;
3. `pattern` is **post-processed**: a `block_sparse` winner is rewritten to `(1000, 6096)`, while `stream_llm` and `vertical_and_slash` return their own `(a, b)` unchanged.

## Task

Implement `solve(q, k, candidates)`:

- `q`, `k`: `(H, N, D)`, **same shape and same dtype**, float32 or float64, `N >= 64`, `D >= 1`;
- `candidates`: a non-empty list/tuple of `(kind, a, b)`; `kind` is one of the three above, `a` and `b` are ints (`bool` does not count).

\(P\) is computed inside: `softmax(q @ kᵀ / √D)` with the causal mask. All internals are float64; `scores` uses the input dtype.

Return a 4-key dict:

- `scores`: `(H, C)` input dtype, entry `(h, c)` = the attention mass kept by candidate `c` for head `h`;
- `best`: `(H,)` int64, the per-head `argmax`, **ties go to the smaller candidate index**;
- `pattern`: `(H, 2)` int64, the post-processed `(vertical_size, slash_size)` of the winner, `(1000, 6096)` when `block_sparse` wins;
- `mask`: `(H, N, N)` bool, the mask of **the winning** candidate of each head.

Raise `ValueError` on: not 3-D, `q`/`k` shape mismatch, differing dtype or a dtype other than float32/float64, `N < 64`, `D < 1`, `candidates` not a non-empty list/tuple, an entry that is not a length-3 list/tuple, unknown `kind`, `a`/`b` not int, `block_sparse` with `a < 1`, and any other kind with `a` or `b` outside `[0, N]`.

## Example

```
q = k = zeros((1, 64, 4), float64)        # H=1, N=64, D=4
candidates = [("stream_llm", 30, 8),
              ("vertical_and_slash", 30, 30),
              ("block_sparse", 8, 1)]

attn[i, j] = 1 / (i + 1)  for j <= i      # q=0, attention uniform over the causal prefix
scores  = [[0.9081807300938489, 0.9975192100819044, 0.0]]
best    = [1]
pattern = [[30, 30]]
mask[0, 63] = cols 0..29 + cols 34..63    # 30 vertical columns + the last 30 diagonals
```

`block_sparse` sees only \(\texttt{block\_num}=2\) blocks, and with `topk_ratio=8` you get `2 // 8 = 0`, so nothing survives — an empty mask and a score of 0. Keeping the same 30 vertical columns but widening `slash` from 8 diagonals to 30 lifts the mass from 0.9082 to 0.9975; the columns 30..33 dropped on the last row are exactly those that lost the vertical contest and were not covered by any chosen diagonal.

## Constraints

- **vertical/slash only see the last 64 queries**: with `N > 64` the earlier rows take no part at all, and only `N = 64` means "every row" — that is where `N >= 64` comes from.
- **The forced keep is "set to \(+\infty\), then sort", not "skip"**: when `a < 30` or `b < 30` those 30 tied \(+\infty\) entries are sorted as well, ties going to the smaller index, so `vertical` starts dropping column 0 and `slash` starts taking the oldest \(\delta=-29\) diagonal.
- **The block mean always divides by 32**: the zeros padded into the last partial block stay in the `mean`, which shrinks the block vector and lowers its block score; renormalizing by the real row count would select different blocks.
- **The quota is "drop the lowest"**: `vertical` drops `N - a` columns and `block` drops `block_num - block_num // a` blocks; with `block_num < topk_ratio` nothing survives.
- **Two different scalings**: \(P\) divides by \(\sqrt D\), the `block_sparse` block dot products do not.
- **`block_sparse` ignores `b` entirely**: it must still be an int, but it is **not range-checked against `[0, N]`**.
- **`mask` is the per-head winner**: not the union of the three candidates; a `stream_llm` mask is identical across heads, the other two are not.
