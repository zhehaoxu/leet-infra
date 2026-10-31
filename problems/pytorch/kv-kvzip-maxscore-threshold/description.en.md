# KV Cache: KVzip Cross-Query Aggregation and Global Threshold Pruning

## Problem

After prefill, KVzip (arXiv 2505.23416) answers one question: **which KV pairs of this context can be dropped**. It never looks at a single query. It spreads out the attention of **every prefill query position**, gives each KV pair a "historical maximum" score, then derives one **global threshold** across layers and heads and evicts every pair below it.

The score comes from `attention/score.py::KVScore._get_score`:

- under GQA each KV head owns \(G\) query heads, and the attention weights are reduced by a max over **the group axis and the query axis at once** — upstream is literally `attn_weights.amax(dim=(-3, -2))` with the comment `max over group, q`;
- each layer, each KV head therefore yields a score vector of length \(N\), and all \(L\) layers are flattened into \(T=L\cdot H_{\text{kv}}\cdot N\) numbers;
- sort them descending, take the value at **index** \(n=\max(\lfloor T\cdot\text{ratio}\rfloor-1,\ 0)\) as the threshold, and keep what is **strictly greater** (`attention/score.py::KVScore._threshold`).

\(n\) is an **index, not a count**: the threshold element itself is always dropped, so the ratio can only under-keep. With `ratio >= 1` upstream keeps everything and returns a threshold of 0. `--level pair-uniform` switches to a per-layer, per-head topk instead (`_threshold_uniform`, \(k=\lfloor N\cdot\text{ratio}\rfloor\)).

## Task

Implement `solve(attn_weights, ratio=0.3, level="pair")`:

- `attn_weights`: `(L, H_kv, G, Q, N)` float32/float64, weights that are **already softmaxed and already stripped of the sink prefix** (sink retention and causal masking are out of scope here);
- `ratio`: a real number `>= 0`, default 0.3;
- `level`: `"pair"` (one global sort and threshold) or `"pair-uniform"` (per-head topk).

Return a 6-key dict (tensors in the input dtype, integers as int64):

- `scores`: `(L, H_kv, N)`, the amax over the group and query axes;
- `valid`: `(L, H_kv, N)` bool, the retention mask;
- `keep_count`: `(L, H_kv)` int64, kept pairs per head;
- `threshold`: Python float, the global threshold (0.0 in the `pair-uniform` and `ratio >= 1` branches);
- `threshold_index`: 0-dim int64, i.e. \(n\) (also 0 in those two branches);
- `keep_ratio`: Python float, the real retention `valid.float().mean()`.

Ties inside one (layer, head) go to the **smaller key index** (upstream's `torch.topk` leaves tie order undefined; this problem fixes it). Raise `ValueError` on: not 5-D, a dtype other than float32/float64, any zero-sized axis, `ratio` not a real number or negative, an unknown `level`.

## Example

```
attn_weights: L=1, H_kv=2, G=1, Q=2, N=2   -> T = 1*2*2 = 4
head 0: query 0 [0.7, 0.3], query 1 [0.4, 0.6]   -> scores[0] = [0.7, 0.6]
head 1: query 0 [0.2, 0.8], query 1 [0.5, 0.5]   -> scores[1] = [0.5, 0.8]

ratio = 0.5, level = "pair"
flat       = [0.7, 0.6, 0.5, 0.8]
descending = [0.8, 0.7, 0.6, 0.5]
n          = max(int(4*0.5) - 1, 0) = 1        # index, not count
threshold  = descending[1] = 0.7
valid      = scores > 0.7 = [[F, F], [F, T]]   # 0.7 itself is dropped
keep_count = [[0, 1]]        keep_ratio = 0.25 # not 0.5

ratio = 0.5, level = "pair-uniform"            # k = int(N*ratio) = 1 per head
valid      = [[T, F], [F, T]]
keep_count = [[1, 1]]        keep_ratio = 0.5
threshold  = 0.0             threshold_index = 0
```

For the same ratio, `pair` keeps a single pair (25%) while `pair-uniform` keeps two (50%): the two conventions do not retain the same fraction.

## Constraints

- **The aggregation is a max**: `amax` runs over the group and query axes together, not a mean and not a sum. A mean flattens the evidence that "one query really cared about this pair", and in long contexts exactly those spikes decide what survives.
- **The threshold is an index**: `n = max(int(T*ratio) - 1, 0)`. The `-1` makes it one fewer than a count; `max(...,0)` is upstream's guard for the degenerate case — when `T*ratio < 1` the threshold is the maximum and **nothing is kept**.
- **Strictly greater**: `valid = scores > threshold`. When every score is equal the threshold is that very value and the result is empty.
- **Global vs per-head**: `pair` sorts all \(L\cdot H_{\text{kv}}\cdot N\) scores together, so a weak head can lose everything; `pair-uniform` gives each head \(\lfloor N\cdot\text{ratio}\rfloor\) slots and heads never compete for budget.
- **Do not add a value-norm term**: there is no `linalg.norm` anywhere in upstream's scoring path; the score is built purely from aggregated attention weights.
- **float64 inside**: upstream's softmax carries a `not fp32` comment and the score keeps that dtype; this problem compares in float64 throughout so that ties near the threshold do not jitter in float32.
