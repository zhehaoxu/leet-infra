# Notes · Q-Filter

## Intuition: importance is learned, not guessed

Every other scorer in the family is a **hand-crafted prior**: knorm bets "big norms matter" (order 146), observed attention bets "being watched a lot matters" (149), expected attention bets "future queries will look here" (150). Q-Filter (arXiv:2503.02812) argues these priors carry systematic bias on QA accuracy and instead learns, per (layer, kv head), a filter \(f\in\mathbb{R}^{D}\) so that \(-f\cdot k\) directly predicts "the answer-loss caused by evicting this key". The filters live in a `QFilters` module (one `nn.Parameter` of shape `(num_layers, num_kv_heads, head_dim)`) hosted per model on the HuggingFace hub and auto-downloaded by `post_init_from_model` — exactly why the offline judge must feed them to `solve` as plain tensors.

## The core three lines

```python
q_filter = self.q_filters[module.layer_idx][None, :, None]   # (1, H_kv, 1, D)
scores = -(q_filter * keys).sum(dim=-1)                      # (B, H_kv, k_len)
```

then the generic ScorerPress flow takes the top-\(n\_kept\). Key points:
- **The minus sign**: score = negated dot product. Top-k of that = the keys with the *smallest* (most negative) raw projection onto \(f\).
- **No scaling**: unlike QKᵀ there is no \(\sqrt{D}\) division; ordering ignores constants, but the returned `scores` must keep raw magnitudes.
- Filters are per (layer, kv head): **no sharing across heads/layers**, no group averaging (kv heads map one-to-one).

## Comparison with neighbouring problems

- `kv-knorm-importance` (146): score = ‖k‖₂, invariant to rotations/basis; Q-Filter scores depend on the inner product with a fixed direction \(f\) and are **tightly coupled to the coordinate system** — RoPE rotation changes the ranking (which is why pairing with KeyRerotation, order 167, matters).
- `kv-expected-attention` (150): also consumes query-side statistics (μ, Σ); Q-Filter collapses "what future queries look like" into one learned vector and needs only keys.
- `kv-think-channel-mask` (153): Think picks **per-token which channels** participate in the sum; Q-Filter applies **one fixed per-head channel-weight vector** to every token — sparse mask vs dense inner product.

## Pitfalls

- Forgetting the minus sign is the main trap: you get the mirrored selection set.
- Tie semantics: a zero filter, or keys orthogonal to \(f\), give equal scores; stable order by (−score, position) keeps the leftmost.
- \(n\_kept\) floor of 1: when \(\lfloor L(1-r)\rfloor = 0\) one position must still be returned.
- `kept` is sorted ascending; `scores` stays in original order.
