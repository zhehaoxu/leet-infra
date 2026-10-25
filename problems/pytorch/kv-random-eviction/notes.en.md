# Notes · Random Eviction

## Why the dumbest baseline must exist

The sneakiest false positive when evaluating KV compression: a scorer correlates with "the last query's attention", which is itself dominated by recency bias — you score well just by keeping recent tokens. `RandomPress` (kvpress `random_press.py`) top-k's on pure uniform scores, giving an **information-free floor**; the standard practice (as seen in LongBench/RULER-style ablations) is to still wrap it with the StreamingLLM head/tail protection, because even the random baseline essentially never evicts sinks/recent — their attention share is so large that keeping them is free. This problem is exactly that composite.

## Engineering the randomness

kvpress reproduces randomness with `torch.Generator(seed)`; the judge goes further — **generators are banned entirely** and the uniform draws arrive as an input tensor. Then (a) the reference and the user code play on identical dice, (b) failing cases replay exactly, (c) no drift between torch `rand` implementations. "Top-q draws" is the sorted equivalent of a uniform without-replacement sample of size q.

Branches most often missed:
- Quota \(q=\max(0,n\_kept-|F|)\): **subtract the forced set first, then lottery**, don't sample the whole sequence.
- \(|F|>n\_kept\): the kept set may **exceed the budget** — StreamingLLM protection outranks the ratio (same "protection over proportion" family flavour as the per-chunk floor in `kv-chunk-independent-topk`).
- The 0/0 fallback for coverage: with all-zero `attn_weights`, coverage degenerates to the positional ratio \(|kept|/L\).

## Comparison with neighbouring problems

- `kv-streaming-llm-sink` (145): deterministic scores "sink+window = 1, middle = 0"; this problem replaces the middle zeros with a lottery — 145 randomized.
- Every informative scorer (146, 149, …): papers must ablate against this baseline — **if the attention-coverage / downstream gap versus random is within noise, the method has no signal**.
- The coverage metric itself is the standard proxy for "reconstruction error" used by SnapKV/PyramidKV evaluations: how much of the last query's attention mass survives eviction.

## Pitfalls

- Ties in `draws` (a deliberately all-equal vector appears in tests): order by (−draw, position); bare `torch.topk` does not guarantee this.
- Overlapping ranges: when \(n\_sink+window\_size\ge L\) the middle is empty and \(F\) covers everything — `kept` has length \(L\), not \(n\_sink+window\_size\) (de-duplicate!).
- Coverage is a scalar float; tests compare with `abs(got-exp)<=1e-5`.
