# KV Cache: Q-Trust Two-Hop Attention Propagation

## Problem

When picking KV entries by attention after prefill, the SnapKV recipe only looks at the attention the last W queries (the request itself) pay to each key. Q-Trust argues that this is one hop only: the **intermediate queries** the request actually attends to should also count towards a key's score. So scoring is split into two hops, with query weights carrying the mass across.

The first hop (`qtrust/reference.py::first_hop`) computes the causal attention of only the last `W` queries and sums it inside each GQA group: \(s(j) = \sum_{t \in \text{last } W} \sum_{h \in \text{group}} A_h(t, j)\). That is a SnapKV score with no pooling.

Query weights (`query_weights`): every request-segment position gets a raw weight of exactly 1, every other position gets `s(q)`, and the whole thing is normalized to \(\sum_q w(q) = 1\). That 1 is a **raw** weight, before normalization — the request is usually a dozen queries and only beats hundreds of document queries because each of its members brings a full 1.

Selective computation (`tau`): after normalization, queries with \(w(q) < \tau / N\) are dropped (`w` set to 0, their attention row never computed), but request-segment queries are always kept; `tau = 0` is the dense second hop.

Second hop (`second_hop`): `score(i) = Σ_{q ≥ i} w(q) Σ_{h∈group} A_h(q, i)`, with the causal mask zeroing every \(q < i\) term. Upstream notes that \(\sum_q s(q) = W \cdot G\), so the result is proportional to \(s(i) + \sum_{q \notin W} s(q) A(q, i)\) — the request's direct attention plus the attention of the queries the request attends to.

Key selection (`select_keys`): each KV head always keeps the last `recent` keys, and the remaining `budget - recent` slots go to the highest `score` among `[0, N - recent)`.

Three things that are easy to get wrong:

1. **Causality is not relaxed at the request boundary**: `q` and `k` are the same N prefill positions, and column \(i\) is visible only to rows \(t \ge i\). The request sits at the tail, so it sees all N keys — but its own rows are invisible to earlier keys.
2. **Normalization happens before dropping**: a dropped query does not hand its weight to the survivors, so `Σq w(q)` can be below 1.
3. **The request is never dropped**: `keep[:, request_mask] = True` overwrites the threshold comparison, so no `tau` can remove a request query.

## Task

Implement `solve(q, k, request_mask, window=16, tau=1.0, recent=16, budget=32)`:

- `q`: `(N, H, D)` float32/float64, NHD layout;
- `k`: `(N, KV, D)` same dtype, \(H = KV \cdot G\);
- `request_mask`: `(N,)` bool, the request segment;
- `window`, `recent`, `budget`: integers; `tau`: a non-negative float;
- the scale is \(D^{-1/2}\); everything inside is float64 and the outputs use the input dtype.

Return a 5-key dict:

- `first_hop`: `(KV, N)`, the first-hop score `s`;
- `query_weight`: `(KV, N)`, `w` after normalization and thresholding (dropped positions are exactly 0);
- `kept_query`: `(M,)` int64 ascending, the union over KV heads of the queries that survive — the attention rows selective computation actually has to run;
- `score`: `(KV, N)`, the second-hop score;
- `selected`: `(KV, budget)` int64, each row ascending: `budget-recent` indices from `[0, N-recent)` by descending score (ties go to the smaller index), plus `N-recent … N-1`.

Raise `ValueError` on: not 3-D, sequence length or head_dim mismatch, differing dtype or a dtype other than float32/float64, `H` not a positive multiple of `KV`, `request_mask` not a `(N,)` bool, `window`/`recent`/`budget` not valid integers, `tau < 0`, `window > N`, `recent > budget`, `budget ≥ N`.

## Example

```
q = ones(6, 1, 1)             # N=6, H=KV=1, D=1
k = zeros(6, 1, 1)            # all logits are 0 -> uniform causal attention
request_mask = [F,F,F,F,T,T]
window = 2                    # the first hop only reads rows 4 and 5
tau = 1.0                     # threshold = tau / N = 1/6
recent = 1, budget = 3        # keep [5], then pick 2 out of [0,5)

A(t, i)      = 1/(t+1) for i <= t     # causal softmax of all-equal logits
first_hop    = [11/30, 11/30, 11/30, 11/30, 11/30, 1/6]
               # j <= 4 gets 1/5 + 1/6, j = 5 gets only 1/6
raw_w        = [11/30, 11/30, 11/30, 11/30, 1, 1]      # request raw weight is 1
sum(raw_w)   = 104/30
query_weight = [0, 0, 0, 0, 30/104, 30/104]
               # 11/104 = 0.1058 < 1/6 -> the first four queries are dropped
kept_query   = [4, 5]
score(i)     = 30/104 * (A(4,i) + A(5,i))
             = [11/104, 11/104, 11/104, 11/104, 11/104, 5/104]
selected     = [[0, 1, 5]]    # five-way tie inside [0,5), take the two smallest
```

`k = 0` collapses the causal softmax to `1/(t+1)`, which is what makes the example hand-computable; note that the first four entries of `query_weight` are 0 — after normalization they are 11/104, just short of 1/6.

## Constraints

- **The request's raw weight is a literal 1**: not a multiple of `s(q)`, and not a constant applied after normalization. Using `s(q)` for every query lets the document queries drown the request by sheer count.
- **The threshold is `tau / N`, and N is the full length**: not the number of non-request queries, and not `window`.
- **The request is never dropped**: an explicit overwrite after the comparison. With a huge `tau` only the request survives, and then `Σw = n_req / Σ_j raw_w(j) < 1`.
- **`kept_query` is the union over KV heads**: each head has its own `w`; as soon as one KV head keeps a query, its attention row (all query heads of the group) has to be computed.
- **The tail is always kept and never competes**: the last `recent` entries of `selected` are always `N-recent … N-1`; with `budget = recent` there are no contest slots and `selected` is exactly that tail.
- **Ties go to the smaller query / key index**: upstream's `torch.topk` leaves tie order undefined, so this problem fixes it; the example and the test both build all-tied cases.
