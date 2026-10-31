# Solution Notes · R-KV Redundancy-Aware Scoring

## In one sentence

The eviction score is **attention minus redundancy**: a token that is attended but duplicated should leave. Four steps are a windowed softmax mean, a max-pool, the pairwise-cosine redundancy with representative zeroing, and a 0.07-weighted fusion followed by topk plus a hard-kept window.

## Why each term

**Attention term.** Only the last `window_size` queries are used — they are the freshest evidence of what matters now; softmax over keys per query and average over queries yields a 0-1 energy distribution. `max_pool1d(k=7)` just smooths neighbouring tokens so a single spike cannot decide the eviction.

**Redundancy term.** `k_norm @ k_normᵀ` with the diagonal cleared (a token is not redundant with itself) and `> 0.5` marked as duplication. A cluster of near-duplicates needs only one representative, so every row zeroes the similarity at its representative and the **column** mean then reads "how many rows consider me redundant"; a softmax turns that into the redundancy score.

**Fusion.** `mix_lambda = 0.07`: attention is worth 0.07 and redundancy 0.93, so the ordering is essentially decided by "how duplicated are you", with attention only breaking ties inside a redundancy tier. That is a deliberate trade-off from the paper: **duplication deserves eviction more than low attention does**.

## What is judged

- `attn` slices keys with `[:-window_size]`: window keys never enter the attention score, yet they **do** enter the cosine matrix and the representative search;
- `redundancy` slices the window columns after the softmax, so its `N` entries sum to less than 1;
- the representative is the max of the hit-index row with non-hits filled by 0, hence 0 for empty rows and hence a nearly constant 0 for `"first"`;
- `final` adds attention and subtracts redundancy with weights `λ` and `1-λ`;
- the first `budget-window_size` entries of `kept` come in descending score order, the last `window_size` in original order, ties to the smaller index.

## Contrast with the neighbours

`kv-merging-cos-nearest` also uses cosine, but there as a **positive** signal for choosing a merge target (more similar means merge them). Here cosine is a **negative** scoring term (more similar means evict), and nothing is merged — only dropped. `kv-h2o-heavy-hitters` has the attention term alone and accumulates it over steps; this problem adds redundancy at token granularity.

## Complexity

Attention scoring is \(O(H W N D)\) and the cosine matrix \(O(H L^2 D)\); at cache length around `budget` (128 by default) the latter is negligible, but on an uncompressed long cache it becomes quadratic — which is why upstream compresses once per step, not per token.

## Source

`Zefan-Cai/R-KV` (arXiv 2505.24133):

- `rkv/compression/r1_kv.py:37-67`: `_compute_scores`; L40-48 softmax over the last `window_size` queries and `mean(dim=-2)`; L51-56 `F.max_pool1d(kernel_size=7, padding=3, stride=1)`; L58-62 `cal_similarity(...)[:, :, :-window_size]`; L64-66 `attn*w - sim*(1-w)`;
- `rkv/compression/r1_kv.py:78-79`: `if kv_cache_len < self.budget: return` (this problem turns that gate into a `ValueError`); L86 `final_score.topk(self.budget - self.window_size)`; L172-183 gather by those indices and concatenate the last `window_size` tokens back;
- `rkv/utils.py:6-30`: `compute_attention_scores`, L12-14 divides by `sqrt(head_dim)`, L23-26 is GQA max-pooling across the query group (this problem fixes `H_q == H_kv` and never takes that path);
- `rkv/utils.py:33-66`: `cal_similarity`; L41 `key/(norm+1e-8)`; L43-44 diagonal `fill 0`; L46 `similarity_cos > threshold`; L55 `torch.max(indices, dim=-1)`; L65 `scatter_(-1, retain, 0)`; L66 `mean(-2).softmax(-1)`;
- defaults at `r1_kv.py:10-15`: `budget=128 / window_size=8 / kernel_size=7 / mix_lambda=0.07 / retain_ratio=0.1 / retain_direction="last"`, plus `assert budget - window_size > 0` at L20.

**Additions of this problem**: ties go to the smaller index (upstream `topk` leaves order undefined); `q` carries only the last `window_size` queries and the absence of a causal mask is explicit; the degenerate `"first"` branch is pinned in the statement; `retain_ratio` (used only by the `*_percent` branches) is out of scope.

## Related problems

- `kv-merging-cos-nearest`: cosine as a positive signal for picking a merge target; here it is a negative signal for eviction.
- `kv-h2o-heavy-hitters`: attention only, accumulated across steps.
- `kv-snapkv-pooling`: real attention plus pooling inside an observation window, no redundancy term.
