# Generative-Agent Memory Retrieval: Three-Factor Weighted Ranking Ledger

## Problem

Generative Agents (the Stanford smallville paper) rank memories with three weighted factors: **recency** (time decay), **relevance** (cosine similarity to the current focal point), and **importance** (poignancy). This problem transcribes the scoring ledger of `new_retrieve`: given a query vector `query_emb` and a list of `memories` (each with `emb` and `importance`), min-max normalize each of the three components to [0, 1], combine them with weights `gw`, and return each memory's score, the component breakdown, and the top `k` indices in stable descending score order.

## Solution

The ledger is pinned by the spec (faithful to the official `retrieve.py`, quirks included):

- `recency[i] = decay ** (i + 1)`, timeline starting at 0: the **oldest** memory has the smallest exponent and hence the largest value (verbatim behavior of the official `extract_recency`; the problem does not "fix" it);
- `relevance[i] = cos(emb_i, query_emb)`: dot product divided by the product of the two L2 norms;
- each component is min-max normalized `(v - lo) / (hi - lo)`; when **range == 0 (including a single node) all values become 0.5** (the official `normalize_dict_floats` picks the midpoint of the target range);
- `score[i] = gw[0]*recency_n[i] + gw[1]*relevance_n[i] + gw[2]*importance_n[i]` with default `gw = (0.5, 3, 2)`;
- `top` takes the first `k` indices in **stable descending** score order (Python's stable `sorted`: ties keep timeline order).

Returns `{"scores", "top", "components": {"recency", "relevance", "importance"}}`.

## Constraints

- 1 ≤ len(memories) ≤ 10^4, 1 ≤ D = len(query_emb) ≤ 64, 1 ≤ k ≤ 10^4.
- Empty query_emb/memories, dim mismatch, non-numeric emb, non-numeric or NaN/Inf importance, non-positive-int or bool k, decay outside (0, 1], gw length ≠ 3 → `ValueError`.
