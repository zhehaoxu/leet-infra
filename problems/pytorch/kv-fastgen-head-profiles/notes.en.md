# Solution Notes · FastGen

## Why classify per head

Heads within one layer behave very differently: some are "delimiter detectors" that pin their mass on a couple of punctuation tokens, some must look back over the whole prompt to aggregate information, and some only look at the last few tokens to keep local fluency. FastGen gives each class a policy that fits it **by construction**, instead of forcing every head through the same top-k.

This is the same lineage as `kv-duo-attention-heads` (retrieval vs streaming, a binary split) but finer: DuoAttention only decides "keep sink+window or not", while this problem adds a specialized class and, more importantly, requires **different budgets per class**.

## Intuition for the two statistics

- \(R_h\), the recency ratio: treat "how much mass sits in the last window" as the streaming tendency. Using the mass fraction of the **last window** rather than an average position is what makes it sensitive to right-skewed distributions.
- \(C_h\), the concentration: the fraction taken by the single largest column. It captures spikes—the signature of a specialized head is dumping mass on very few tokens. Using max instead of entropy keeps the threshold interpretable and hand-checkable.

Neither needs sorting: one reduction is enough, so both are natural for a streaming profiler.

## Why streaming is tested first

If `R` qualifies, the head's attention is almost entirely in the recent window, and sink + window already covers its needs; a scoring policy would be redundant. If specialized were tested first, a head that is both spiky and recency-dominated would be dragged into top-k and lose the cheaper positional policy. The evaluation order is part of the algorithm, and the statement fixes streaming first.

## Positional vs scoring policy

The streaming branch is the easiest thing to get wrong in this problem: it **never looks at the scores**. Even if the last window is all zeros and the middle holds a huge spike, the last window stays and the spike goes. That is not a bug but the point—a streaming head needs positional continuity, and scoring would break the integrity of its local syntactic window.

## Ties and boundaries

Top-k is "descending score, ties to the smaller index", consistent with the rest of the library; `k` may exceed the candidate count, in which case everything is taken. All threshold comparisons use **`>=`**, so the hand-computed boundary cases land exactly on the class they test.

## Complexity

\(O(HL)\) for the statistics plus \(O(HL\log L)\) for the per-head sort, with \(O(HL)\) space for the output mask. A real implementation parallelizes over heads, giving \(O(L\log L)\) per head.

## Related problems

- `kv-duo-attention-heads`: binary classification with fixed sink/window, no budget differentiation.
- `kv-pyramid-layer-budget`: allocates budgets across **layers**; this problem allocates across **heads**, so they are orthogonal.
- `kv-streaming-llm-sink`: the pure positional baseline, i.e. this problem's streaming branch.
