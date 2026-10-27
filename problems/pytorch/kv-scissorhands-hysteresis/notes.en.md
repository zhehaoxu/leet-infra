# Solution Notes · ScissorHands

## Why hysteresis is needed

Single-frame pruning treats importance as i.i.d. observations, but decoding-time attention **jitters from frame to frame**: one formatting token appearing can pull a whole block of attention mass away. With per-frame top-k, a wrongly killed key token is gone forever. ScissorHands' hysteresis adds a "confirm over \(m\) consecutive frames" buffer: if any frame recovers, the counter resets and the token survives.

The price is **reaction latency**: real staleness only takes effect at the \(m\)-th frame. That is the `patience` trade-off—larger \(m\) is more stable but slower, and \(m=1\) degenerates to memoryless threshold pruning.

## Three traps in the counter

1. **Strictly less than**: `score < tau` is a low score. Using `<` rather than `<=` matters, and the `score == 0.5` with `tau = 0.5` case separates the two.
2. **Reset on recovery**: this is the **consecutive** low count, not the cumulative one. Any frame with `>= tau` zeroes the counter; otherwise a token with sparse noise would slowly accumulate its way to eviction.
3. **Update before testing**: the current frame's score is counted first, then the threshold is checked. Testing first shifts the eviction step by one.

## Irreversible eviction

The implementation only writes an eviction step when `evict_step < 0` (the `& alive` term). In a real system an evicted token has already left the cache and has no meaningful later score; allowing resurrection would turn hysteresis into a sliding-window average with completely different semantics.

## Division of labour with single-frame scoring problems

The existing `kv-snapkv-pooling`, `kv-knorm-importance` and `kv-tova-head-mean` problems all compute **one** score pass plus top-k—they decide *who*. This problem adds no new scorer and focuses on *when* a token is actually deleted: decision filtering along the time axis, orthogonal to the choice of scorer.

## Complexity

\(O(SL)\) time and \(O(L)\) extra space: each step is one elementwise comparison, one counter update and one threshold test. No sorting, no need to store the full score history—it is fully streaming.

## Related problems

- `kv-snapkv-pooling` / `kv-knorm-importance`: single-frame scoring and selection (the upstream of this problem).
- `kv-streaming-llm-sink`: purely positional rules, no temporal decision.
- `kv-h2o-heavy-hitters`: memory lives in the score; here memory lives in the **decision**, so the two compose.
