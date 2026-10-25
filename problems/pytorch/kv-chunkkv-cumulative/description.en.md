# ChunkKV: Whole-Segment Retention by Semantic Chunk

## Problem

**ChunkKV** (Xu et al., 2025, [arXiv:2502.00297](https://arxiv.org/abs/2502.00297)): unlike ChunkPress, which "picks tokens inside each chunk", ChunkKV cares about **semantic coherence of the whole chunk**—it aggregates a run of consecutive tokens into one chunk, computes a single **chunk-level score**, then keeps the top-k chunks by that score **as a whole or drops them as a whole**, avoiding "picking a few tokens out of a chunk and breaking the meaning".

## Rules

You are given the per-head token `scores` of shape $(H, L)$; a positive integer `chunk_length`; and a float `compression_ratio` $\in[0,1)$:

1. Cut the length dimension into $N=\lceil L/c\rceil$ segments (the last may be shorter than $c$).
2. **Sum across heads, then average within the chunk** to get the chunk-level score:
   - For each token $t$ first compute $s_t=\sum_h \text{scores}_{h,t}$ (sum across heads).
   - Chunk $i$ covers $[ic,\min((i+1)c,L))$, and $cs_i = \text{mean}_{t\in \text{seg}_i}(s_t)$ (**divided by the chunk's actual length**).
3. Number of chunks kept $n\_chunks\_kept = \max\big(1,\ \lfloor N \cdot (1-\text{compression\_ratio})\rfloor\big)$.
4. Take the top $n\_chunks\_kept$ of $cs$ in **descending** order (ties: smaller chunk index wins).
5. **Merge the token positions of all selected chunks** and output them **sorted by ascending position**. Chunk selection is shared across heads (because step 2 already summed the head dimension away).

Return a 1-D `long` tensor of shape $(n\_tokens\_kept,)$.

## Examples

```
H=2, L=12, chunk_length=4, compression_ratio=1/3
head0 scores: [0.5, 0.4, 0.6, 0.5, 0.05, 0.05, 0.05, 0.05, 0.4, 0.3, 0.5, 0.4]
head1 scores: [0.4, 0.5, 0.5, 0.4, 0.05, 0.05, 0.05, 0.05, 0.3, 0.4, 0.4, 0.3]
Sum across heads: [0.9, 0.9, 1.1, 0.9 | 0.1, 0.1, 0.1, 0.1 | 0.7, 0.7, 0.9, 0.7]
Chunk means: cs = [0.95, 0.1, 0.75]
N=3, n_chunks_kept = max(1, floor(3·2/3)) = 2 → top2 = chunk0, chunk2
kept tokens = {0,1,2,3} ∪ {8,9,10,11} = [0,1,2,3,8,9,10,11]
```

## Constraints

- CPU float32; chunking happens only along the $L$ dimension; $c\ge 1$.
- With $N=1$ ($L\le c$) the single chunk is always kept → all positions.
- Ties are sorted stably by $(-cs_i,\ i)$.
