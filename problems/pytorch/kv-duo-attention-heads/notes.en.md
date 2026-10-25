# Solution Notes · DuoAttention

## Key Insight: Not Every Head Needs the Full KV

The authors found that attention heads in large models are highly specialized: a few **retrieval heads** do the work of "fishing information back out of a very long context", and their attention really spreads to distant positions; many **streaming heads** attend in a highly local way—they only see the attention sink at the start of the prompt and the recent window. Given that, keeping the full KV for the latter is pure waste.

DuoAttention therefore compresses **along the head dimension**: streaming heads are downgraded to "keep only sink+recent" (essentially per-head StreamingLLM), while retrieval heads stay full. Mixing the two gives "a higher compression ratio at the same quality".

## How Head Type Is Decided

In a real system the head type is estimated from dataset statistics (per head, compute the "area under the cumulative attention curve"; the closer to 1, the more retrieval-like). This problem **externalizes** that step as the input `head_scores`—higher scores mean more "retrieval", and the smallest $n\_pruned$ heads are downgraded to streaming. That keeps the exercise focused on the "split mask" step without reproducing the profiling pipeline.

## Relation to StreamingLLM

A single streaming head's mask `[T,T,F,F,F,F,T,T]` is exactly the StreamingLLM sink+window structure. DuoAttention = "**per head**, apply StreamingLLM at selected positions in certain layers and leave the other heads untouched". Put another way: StreamingLLM streams *all* heads, DuoAttention streams only the heads classified as streaming.

## Ties and Rounding

- `n_pruned = floor(H·ratio + 0.5)` states clearly that half rounds up, not banker's rounding.
- When head scores tie, the smaller index enters the streaming set first—both oracle and implementation sort stably by ascending `(score, index)` to avoid flaky results from platform-dependent `argsort` behaviour.

## Overlapping Windows Boundary

When `sink_size + recent_size >= seq_len`, the streaming head's head and tail windows cover the whole row, so the mask is naturally all `True`. In the implementation you only carve out the middle `if L > S+R`; `mask[h, S:L-R]` is an empty slice when $S+R\ge L$, so the two formulations are equivalent—this problem keeps the explicit guard because it is clearer.
