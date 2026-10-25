# ChunkPress: Per-Chunk Independent Top-K Retention

## Problem

**ChunkPress** (from FINCH, Zhu et al., 2024, [arXiv:2407.09112](https://arxiv.org/abs/2407.09112)) is a "wrapper"-style compression method: a plain ScorerPress runs one top-k over the whole sequence, which tends to pile all retention slots into a few high-scoring spans. ChunkPress cuts the sequence into **equal-length chunks** and **takes top-k inside each chunk separately**, which guarantees the compression spreads evenly over the whole context and misses no span.

## Rules

You are given the per-head token importance `scores` of shape $(H, L)$; a positive integer `chunk_length`; and a float `compression_ratio` $\in[0,1)$:

1. Cut $L$ along the length dimension into segments: segment $i$ covers positions $[ic,\ \min((i+1)c, L))$, where $c=$ `chunk_length`. **The last segment may be shorter than $c$**.
2. For each segment, the number kept is
   $$n\_kept = \max\big(1,\ \lfloor \text{len}_{seg}\cdot(1-\text{compression\_ratio})\rfloor\big)$$
   where $\text{len}_{seg}$ is the segment's **actual length**.
3. Within that segment take the top $n\_kept$ by score (higher score wins) and return the **global indices** of those positions (i.e. index within the segment + segment start offset).
4. Concatenate the segments' results **in segment order**; within one segment order them **by descending score**.

Output a `long` tensor of shape $(H, n\_total)$, where $n\_total$ is the sum of the per-segment kept counts.

## Examples

```
H=2, L=8, chunk_length=4, compression_ratio=0.5
head 0 scores: [0.1, 0.9, 0.3, 0.2 | 0.8, 0.05, 0.7, 0.6]
  chunk0 len=4, n_kept=2, top2 = pos 1 (0.9), pos 2 (0.3)
  chunk1 len=4, n_kept=2, top2 = pos 4 (0.8), pos 6 (0.7)
head 1 scores: [0.5, 0.4, 0.9, 0.1 | 0.05, 0.7, 0.6, 0.8]
  chunk0 top2 = pos 2 (0.9), pos 0 (0.5)
  chunk1 top2 = pos 7 (0.8), pos 5 (0.7)

out = [[1, 2, 4, 6],
       [2, 0, 7, 5]]
```

## Constraints

- CPU float32; each row of `scores` is handled independently, with no aggregation across heads.
- With `compression_ratio` $=0$ a segment keeps all of itself; as it $\to 1$ each segment keeps at least 1.
- Ties: equal scores mean **the smaller position wins** (stable, sorted by $(−\text{score},\ \text{pos})$).
