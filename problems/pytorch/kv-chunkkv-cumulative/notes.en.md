# Solution Notes · ChunkKV

## Semantic Chunks: Why Keeping Whole Chunks Is Better

Token-wise scoring (Knorm/SnapKV/TOVA and the like) picks out "high-score points" scattered through the context, but in language **the meaning of a passage usually spans several consecutive tokens**: keeping only three or five tokens of a sentence chops the context up and breaks coherence. ChunkKV's observation is that treating a run of $c$ consecutive tokens as one semantic chunk and deciding **chunk-wise by importance** yields retained KV that reads more like an abridged copy of the original.

## The Two-Step Aggregation of the Chunk Score

$$cs_i = \frac{1}{|\text{seg}_i|}\sum_{t\in \text{seg}_i}\ \sum_{h} \text{scores}_{h,t}$$

First **sum across heads** (combining the opinion of every head), then **average within the chunk** (divide by the chunk length so a short tail chunk is comparable to a full one). This is a completely different path from ChunkPress's "per-chunk top-k": ChunkPress keeps tokens, ChunkKV keeps whole chunks.

## Output Is Ascending and Shared Across Heads

Because chunk selection already aggregates the head dimension away, all heads use the same set of chunks, so the result is naturally a 1-D list of positions. Finish with `.sort()` to get ascending order, so gather reads them in original text order (the relative order of the KV must not be disturbed).

## Safeguards and Boundaries

- `max(1, int(...))` guarantees at least one chunk survives, so even `compression_ratio→1` never empties the set.
- When $L\le c$ there is a single chunk → keep everything. This problem unifies the "full chunks + tail chunk" cases as `ceil(L/c)` segments, each averaged by its actual length; that is equivalent to the official branching code but more concise.

## Orthogonality to PyramidKV / AdaKV

PyramidKV/AdaKV decide "how much budget each layer/head gets", while ChunkKV decides "under the same budget, is the smallest retained unit a chunk or a token". They can be stacked: score with some scorer first, then use ChunkKV to aggregate to chunk-level selection.
