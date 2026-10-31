# Solution Notes · Q-Trust Two-Hop Scoring

## In one sentence

The first hop scores every key with the attention of the last `W` queries; that score then becomes the **credibility** of every other query and drives a second, weighted hop; request-segment queries carry a raw weight of 1 and can never be dropped.

## What the two hops actually compute

Write \(G=H/KV\) and \(A^{\text{g}}_q(i)=\sum_{h\in\text{group}}A_h(q,i)\). The first hop is by definition \(s(i)=\sum_{q\in W}A^{\text{g}}_q(i)\), hence

\[
\text{score}(i)=\frac{1}{Z}\Bigl[s(i)+\sum_{q\notin W}s(q)\,A^{\text{g}}_q(i)\Bigr],\qquad Z=\sum_{q\notin W}s(q)+|W|
\]

The first term is the request's own attention (normalized by a weight that counts each request query as 1), the second is what the queries the request trusts contribute on top. Upstream's remark \(\sum_q s(q)=W\cdot G\) is exactly this identity. Careful: the proportionality **only holds while no query is dropped** — dropped queries are zeroed but not renormalized, so the dropped terms simply disappear from the right-hand side.

## Five conventions that are easy to get wrong

1. **The request's raw weight is 1**, not \(s(q)\): upstream writes `raw[:, N - W:] = 1.0` before normalizing.
2. **The threshold is `tau / N`** — N is the full sequence length, not the number of non-request queries and not `window`; the comparison is `>=`.
3. **The request is never dropped**: `keep[:, N - W:] = True` is an overwrite **after** the comparison, so an arbitrarily large `tau` drops every document query and leaves the request untouched.
4. **`kept_query` is a union**: every KV head computes its own `w`, but a query's attention row is shared by all query heads of the group, so one surviving head is enough (upstream's `keep.any(dim=0)`).
5. **Ties go to the smaller index**: `torch.topk` leaves tie order undefined; both the example and the tests build fully tied cases.

## What is judged

- `first_hop` is a within-group sum and \(\sum_j s(j)=W\cdot G\) always holds (each attention row has mass 1);
- `query_weight` sums to 1 when nothing is dropped, and dropped positions are exactly 0;
- causality of `score`: with no surviving \(q\ge i\) the value at `i` must be 0; the last `recent` entries of `selected` are always the tail, contest slots only come from `[0, N-recent)`, and every row is ascending and duplicate-free.

## Boundaries

- `budget = recent`: no contest slots, `selected` is just the tail; `budget >= N` raises (upstream `select_keys` returns `None` when `N <= budget`, i.e. evicts nothing).
- `window = 1` degenerates the first hop to the group sum of the last row; `window = N` degenerates it to a column sum.
- `tau = 0` is the dense path: every query participates and `Σw = 1`.
- With every non-request query dropped, `Σw = n_req / Σ_j raw_w(j) < 1` and `score` is *not* a renormalized request-only attention.

## Complexity

The first hop costs \(O(H\,W\,N\,D)\) and the second \(O(H\,|Q_{\text{kept}}|\,N\,D)\), against \(O(H N^2 D)\) dense. Selective computation saves the \(|Q_{\text{kept}}|/N\) factor; upstream folds it into the FlashAttention-2 LSE (`qtrust/kernels.py`), while this problem only grades the scoring convention.

## Source

`qtrust/` in `Dirac-Notation/A2SF` (the plain-PyTorch reference for Q-Trust; the repo has no README and neither code nor comments carry an arXiv id, so only the repo and the code paths are cited):

- `qtrust/reference.py` L1-27: the docstring is the specification;
- L31-40 `_attention_rows`: the `cols > rows` causal mask plus softmax;
- L43-50 `first_hop`: `A.view(KV, H // KV, W, N).sum(dim=(1, 2))`;
- L53-73 `query_weights`: L63-64 `raw[:, N - W:] = 1.0`, L67 normalization, L70 `keep = w >= tau / N`, L71 the request override, L72 `w = w * keep`;
- L76-90 `second_hop`: blocked `einsum("kq,kgqn->kn")`, skipping blocks whose weights are all zero;
- L100-109 `select_keys`: L104 returns `None` when `N <= budget`, L106 `head = N - recent`, L107 `topk(budget - recent)` over `score[:, :head]`, L108-109 appends `arange(head, N)`;
- `qtrust/fast.py` L17-19 and `kvcompress/scorers/twohop.py` L72-80: `keep.any(dim=0)` yields the query rows that actually have to be computed.

**Additions of this problem** (not written upstream, pinned explicitly): the request segment is an explicit `request_mask` instead of "the last W queries"; `kept_query` is returned as the union; ties go to the smaller index; shapes and hyper-parameters are validated with `ValueError`. The same repo's `kvcompress/` holds AdaSelector, a re-implementation of Ada-KV that would collide with the existing `kv-adakv-headwise`, so this problem takes only the `qtrust/` two-hop path.

## Related problems

- `kv-quest-page-bound`: page-level attention bounds for page selection, with no query weighting at all; this problem is token-level and its scores come from the two-hop propagation.
- `kv-snapkv-pooling`: a single hop (real attention inside an observation window); the second hop is the whole point here.
- `kv-adakv-headwise`: cross-head budget allocation from the `kvcompress/` line, unrelated to this problem.
