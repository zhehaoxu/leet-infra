# Solution Notes · MInference Sparse Pattern Search

## In one sentence

Turn "which sparse shape should this head use" into a **scoring problem**: measure how much probability mass each candidate mask keeps under one real causal attention \(P\), then take the per-head `argmax`.

## Why kept mass is a usable objective

Each row of \(P\) is a probability distribution, so \(\sum_j P_{ij}M_{ij}\) is the fraction of row \(i\)'s attention kept by \(M\); averaging over rows gives a number in \([0,1]\). It needs no attention kernel and no actual sparse run — a mask you can build is a mask you can score. That is exactly why upstream can settle a pattern with one search pass.

The three masks come from different places:

- `stream_llm`: purely analytic — the first \(a\) columns unioned with the last \(b+1\) diagonals, identical for every head;
- `vertical_and_slash`: uses the last 64 queries of \(P\) as a **proxy** — `vertical[j]` sums the window column-wise, and entry \(j\) of `slash` is the diagonal with slope \(\delta=j+1-N\); whatever carries mass gets kept;
- `block_sparse`: does not even ask \(P\) — it takes block-mean \(q\) and \(k\) and runs one \((N/32)^2\) block-level dot product to guess which blocks matter.

## Conventions that are easy to get wrong

1. **The forced keep is \(+\infty\) fed into the same sort.** `vertical[:30]` and `slash[-30:]` are set to \(+\infty\) so top-k can never drop them; the price is that those 30 entries **tie with each other**: with `a < 30` column 0 is the first to be dropped, and with `b < 30` the oldest \(\delta=-29\) diagonal is the first to be taken. Skipping the forced entries instead of sorting them would drop a column by its real `vertical` value once `a < 30` pushes the drop count past the 34 non-forced columns; `check_vertical_force_keeps_first_30` asserts that column 0 is the one that goes.
2. **The block mean divides by 32, not by the real row count.** The padded zeros stay inside the `mean`, so the last partial block vector is shrunk proportionally and its block score drops. The test builds a \(N=250\) case (last block holds only 26 rows) where the two conventions pick different blocks.
3. **`slash` is a union.** A column that lost the vertical contest still survives if it lies on a chosen diagonal; writing an intersection yields a different mask and a different mass accounting.
4. **The quota counts drops.** `topk(-vertical, N - a)` and `topk(-qk, block_num - block_num//ratio)` both compute how much to discard first, so `block_num < ratio` means every block is discarded.

## What is judged

- `scores` per head and per candidate, with \(C=\texttt{len(candidates)}\), in the input dtype;
- `best` is the row-wise `argmax` with **ties going to the smaller candidate index** — the test pins this with two candidates both scoring 1.0;
- `pattern` carries sizes only, never the kind: a `block_sparse` winner gives `(1000, 6096)`, a `stream_llm` winner gives its own `(a, b)`;
- `mask` must be the mask of **the candidate that won** for that head, `(H, N, N)` bool;
- \(P\) and every intermediate run in float64, `q`/`k` are never modified in place, and non-contiguous views must work.

## Boundaries

- `N < 64` raises: the vertical/slash statistics window is fixed at the last 64 queries.
- `a` and `b` endpoints: `vertical_and_slash(0, 0)` gives an empty mask and score 0, `stream_llm(0, 0)` leaves only the main diagonal, and `a = N` drops no column so the mask is exactly the causal lower triangle.
- `block_sparse`: `topk_ratio = 1` drops nothing and degenerates to plain causal (score 1); `block_num < topk_ratio` drops everything (score 0).
- A fully masked row is not renormalized: with `a = 29` and `b = 1` row 0 is empty (it can only see column 0, and the single chosen diagonal is \(\delta=-29\)), so it just feeds a 0 into the mean.

## Complexity

\(P\) itself costs \(O(HN^2D)\) time and \(O(HN^2)\) memory, so the search phase is not cheap — upstream runs it once per head inside the `is_search` branch and dumps the result into a config file (L555-558 and L570); the savings come afterwards, when long-context inference honours the chosen pattern. By comparison the vertical/slash statistics are \(O(H\cdot 64\cdot N)\), the `block_sparse` block scoring is only \(O(H(N/32)^2D)\), and scoring one candidate is \(O(HN^2)\).

## Source

`microsoft/MInference` (arXiv 2407.02490), `minference/modules/minference_forward.py::search_pattern` (L129):

- **vertical_and_slash**: L134 `last_q = 64`; L136-138 build `qk = matmul(q[…], k.transpose(2, 3))/math.sqrt(head_dim) + attention_mask[…]` over the last 64 queries only, then softmax; L139 `vertical = qk.sum(-2, keepdim=True)`, L140 `vertical[...,:30] = torch.inf`, L141 `torch.topk(-vertical, q_len - vertical_size)`; L143 `slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]` (helper L110-116 sums diagonals with `as_strided`), L144 `slash[...,-30:] = torch.inf`, L146 `torch.topk(slash, slash_size, -1).indices - (q_len - 1)`;
- **stream_llm**: L163-164 `torch.triu(torch.tril(ones, 0), -slash_size)` then `mask[:, :vertical_size] = 1`, followed by `torch.tril` at L167;
- **block_sparse**: L173-179 `block_num = (q_len - 1) // 32 + 1`, zero-pad then `.reshape(…, block_num, 32, -1).mean(-2)`; L181 block-level `matmul` plus the block-causal mask (with **no** `/sqrt(d)` here); L183 `torch.topk(-qk, block_num - block_num//topk_ratio)`; L187-188 `repeat(1,1,1,32,1,32)` expands to a token grid, then `tril` and a `[:q_len, :q_len]` slice;
- scoring at L156-157, L168-169 and L190-191: `attn_weights_x = attn_weights * est_attn`, then `.sum(-1).mean(-1)`; the candidate grids are the hardcoded `vs_list` at L209/L211/L213, and L218 `if score > best_score` uses a strict comparison, so an earlier candidate survives a tie;
- L222-225 post-processing: `stream_llm → vertical_and_slash`, and `block_sparse → vertical_and_slash` with `best_v, best_s = 1000, 6096` — this is where the `pattern` rewrite comes from.

**Additions of this problem** (not written upstream, pinned explicitly): the score averages over **all** query rows, while upstream only counts rows after `attn_weights_x[:,:,2500:]`; the candidate grid is passed in as `candidates` instead of the hardcoded `vs_list`; ties always go to the smaller index: at the candidate level this still matches the strict `>` of L218, but the column, diagonal and block levels have no order that `torch.topk` guarantees, so the convention is the exercise's own; masks and scores are returned per head, whereas upstream calls the search once per head and only records `(kind, v, s, score)` into a config plus `print`s a line; `block_sparse` still takes a triple `(kind, a, b)` whose `b` is neither used nor validated; \(P\) and all intermediates are float64 while upstream softmaxes in float32.

## Related problems

- `kv-streaming-llm-sink`: the StreamingLLM sink + sliding-window KV version; here the same shape is only one candidate to be scored.
- `kv-quest-page-bound`: also "estimate a cheap score, then select", but Quest re-selects every decode step at page granularity; here one search fixes the whole pattern.
- `kv-chunk-independent-topk`: pure top-k bookkeeping; here top-k sits behind the tie trap created by the \(+\infty\) forced keeps.
