# Matryoshka Embedding Truncation: Low-Dimension Rank-Preservation Ledger

## Problem

Embeddings trained with Matryoshka Representation Learning (MRL, arXiv:2205.13147) have a nesting
property: **the first d dimensions are already a usable low-dimensional embedding** — retrieval can
truncate 1024 dims down to 64 for a coarse pass, then re-rank at full dimension, saving an order of
magnitude in storage and compute. This works only if "truncate + **re-normalize** + cosine
similarity" keeps the low-dimension ranking close to the full-dimension one. This problem pins that
retrieval ledger into a judgeable deterministic form: given query and document embeddings, compute
per-truncation similarity, ranking, and the overlap with the full-dimension top-k.

> To keep judging **deterministic**: embeddings are all explicit inputs (no model loading); compute
> in float64 before normalizing; zero vectors normalize to all-zero (norm floor 1e-12); ties break
> by ascending index.

## Rules

- Implement `solve(query_emb, doc_embs, dims, top_k)`:
  - `query_emb: torch.Tensor [D]`, `doc_embs: torch.Tensor [N, D]`;
  - `dims: list[int]`, truncation levels (each `1 ≤ d ≤ D`, may repeat, returned in input order);
  - `top_k: int ≥ 1` (`> N` raises `ValueError`).
- For each dimension `d` (plus the full-dimension `full` level at `d = D`):
  1. `q_d = normalize(query_emb[:d])`, `docs_d = normalize(doc_embs[:, :d])` — **re-normalize after
     taking the first d dims** (the core MRL convention; skipping it degrades cosine to dot product);
  2. `sims = docs_d @ q_d` (cosine similarity);
  3. `top =` the first `top_k` indices sorted by `(-sim, index)`.
- The `full` level uses `d = D` with the same recipe; each truncation level records
  `overlap_full = |top_d ∩ top_full|` (**set** intersection, order-insensitive).
- Return a dict: `full` (`sims` list + `top` list), `truncated` (mapping `d` → that level's ledger
  dict with `sims` / `top` / `overlap_full`), and `n_docs`.
- Validation (`ValueError`): `top_k` not a positive integer (bool included) or `> N`; `d` not an
  integer (bool included), `< 1`, or `> D`; `query_emb` / `doc_embs` dimension mismatch.

## Example

```
Input: query_emb = [1, 1] (45°)
       doc_embs = [[2, 0.5], [1, 2], [-1, 1], [0.5, 0.1]], dims = [1, 2], top_k = 3
  full (d=2): sims ≈ [0.857, 0.949, 0.0, 0.829] → top = [1, 0, 3]
  d=1: only the first dim survives; magnitudes vanish under normalization, sims = [1, 1, -1, 1] → top = [0, 1, 3]
  d=1 overlap_full = 3 (the top sets coincide)
Output: full.top = [1, 0, 3], truncated[1].top = [0, 1, 3], truncated[1].overlap_full = 3
```

## Constraints

- float32 inputs, float64 internal computation (comparison tolerance 1e-6)
- `dims` may contain `D` itself (that level's ledger equals `full`); may be unordered or repeated
- Zero vectors (all-zero rows after truncation) get similarity 0 and join tie ordering
- Returned `sims` are Python float lists (after `.item()`), in `doc_embs` row order
