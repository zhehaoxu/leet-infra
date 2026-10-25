# Solution Notes · ChunkPress

## Core Motivation: Even Coverage vs Concentrated Retention

A plain ScorerPress does one global top-k over the whole sequence. If the score distribution has obvious "high grounds" (say a few spans with much stronger attention), the global top-k hands nearly all slots to those high grounds, and other spans earlier or later in the context may keep not a single token—which is fatal for tasks that need long-range retrieval.

ChunkPress takes a plain approach: **first cut the sequence into segments of fixed length, then pick each segment's own top-k inside it**. Each segment keeps at least 1 (`max(1, ...)`), so no matter how bad the scores are, every span has a representative in the retained set, giving "compression that is uniform along the context".

## Why n_kept Per Segment Uses the Actual Length

The tail of a sequence is often shorter than a full chunk. The formula uses $\text{len}_{seg}$ (the segment's actual length) instead of a fixed $c$, so a short tail segment is compressed at the same ratio and still keeps at least 1.

## The Ordering Pitfall

The retained indices are not globally sorted; the order is **"segment order + descending score within each segment"**. Positions may look "out of order" (like head1's `2,0` in the example), because within a segment the order is by score. Judge segment by segment—do not sort first and then compare.

## Difference from ChunkKV

- **ChunkPress**: top-k **per token** inside each segment (keeps the best few tokens of the segment).
- **ChunkKV**: first aggregate a whole segment into **one chunk score**, then keep/drop chunks as a unit (complete segments are kept, so semantics stay more coherent).

Both use the "chunking" idea but keep different granularities: ChunkPress keeps tokens, ChunkKV keeps chunks.

## Tie Handling

kvpress uses `topk`, whose tie order depends on the implementation. This problem pins the rule to "equal scores mean the smaller position wins": both the oracle and the implementation sort stably by $(-score,\ pos)$ to avoid flaky results.
