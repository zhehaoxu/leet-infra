# Automatic Prefix Caching: KV Block Prefix Hits

## Problem

Many requests share the same prefix: the same system prompt, the history of a multi-turn conversation, a reused few-shot template. Recomputing the KV of that prefix for every request is pure waste. **Automatic Prefix Caching** (vLLM) caches KV cache blocks by content: as long as some **complete prefix block** of a new request is exactly identical to one computed before, it can be reused directly, skipping prefill.

Reuse is **block-granular** and requires a **contiguous prefix**: whether block $j$ can be reused depends on whether "from the start through block $j$" — the whole prefix — matches some previously seen identical prefix chain in the cache (identical content but a different preceding block cannot be shared, because KV depends on position and history). Once one block mismatches, all blocks after it cannot be reused. A trailing **partial block** that does not fill a whole block **does not take part** in cache matching.

## Rules

Given `block_size` (int) and `requests` (`list[list[int]]`, each request's token id sequence, processed in arrival order). Maintain a **prefix-chain cache** that starts empty:

For each request:

1. Take its **full blocks**: $n_\text{full} = \lfloor \text{len}/$`block_size`$\rfloor$ blocks, where block $j$ is `req[j*block_size : (j+1)*block_size]`
2. Build the **cumulative prefix key** $K_j$ of each block = the tuple obtained by concatenating the tokens of blocks $0$ through $j$ (i.e. the key itself encodes the whole prefix — both position and content must match)
3. **Number of hit blocks**: starting from $j=0$, if $K_j$ is already in the cache then $+1$ and continue; on the first $K_j$ **not** in the cache, stop immediately (nothing after it can hit)
4. Regardless of hits, add **all** $K_j$ of this request to the cache (so later requests can share them)
5. Record the number of hit blocks for this request

Return `list[int]`: the number of hit blocks for each request (length = number of requests).

## Examples

```
block_size = 4
requests = [
  [1,2,3,4, 5,6,7,8, 9,10],          # full blocks A=[1,2,3,4], B=[5,6,7,8] (tail 9,10 partial, not counted)
  [1,2,3,4, 5,6,7,8, 11,12,13,14],   # full blocks A, B, C=[11,12,13,14]
  [7,7,7,7, 8,8,8,8],                # full blocks D, E
  [1,2,3,4, 9,9,9,9],                # full blocks A, F=[9,9,9,9]
  [1,2,3,4, 5,6,7],                  # only full block A (tail 5,6,7 partial, not counted)
]

r0: A not cached → 0 hits; cache {A,AB}
r1: A hit, AB hit, ABC miss → 2 hits; cache += {ABC}
r2: D miss → 0 hits
r3: A hit, AF miss → 1 hit
r4: A hit → 1 hit

hits = [0, 2, 0, 1, 1]
```

## Constraints

- CPU; `block_size >= 1`; token ids are integers.
- Cache keys encode the whole prefix (cumulative), so "same content but different prefix" is not a hit.
- Trailing tokens that do not fill a block are excluded from $n_\text{full}$, do not participate in matching, and are not cached.
- Return a Python `list[int]`.
