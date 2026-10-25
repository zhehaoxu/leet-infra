# LRU Eviction for KV / Prefix Cache

## Problem

GPU memory is limited, but the KV blocks of hot prefixes (system prompts, multi-turn conversation history, shared few-shot templates) are worth keeping as long as possible. **LRU (Least-Recently-Used)** is the most common cache eviction policy in serving: when capacity is full, evict the block that has **not been accessed for the longest time**. A hit block is "kept alive" — refreshed as most recently used — so hot data resides for a long time while cold data naturally sinks to the bottom and is evicted. vLLM's prefix caching and the KV block pool reclamation of various inference gateways all use the LRU family as their skeleton.

This problem provides an access stream (one block id per access) and asks you to simulate an LRU cache with capacity `capacity`, count the hits/misses, and give the recency ordering of the blocks in the cache at the end.

## Rules

Given `capacity` (int, the maximum number of distinct blocks the cache can hold) and `accesses` (1-D, the stream of block ids in time order):

Maintain an ordered cache with **head = most recently used (MRU)** and tail = least recently used (LRU). Process each access `a` one by one:

1. **Hit** (`a` is already in the cache): `hits += 1`, move `a` to the head (refresh as most recently used)
2. **Miss** (`a` is not in the cache): `misses += 1`
   - If the cache is full (element count `== capacity`): first remove the **tail** (LRU)
   - Insert `a` at the head
3. When `capacity == 0` the cache is always empty, every access is a miss, and nothing can reside

Return the triple `(hits, misses, final_order)`: `hits`/`misses` are ints and `final_order` is `list[int]`, giving the cache contents after all accesses, **ordered MRU→LRU** (head first).

## Examples

```
capacity = 3
accesses = [1, 2, 3, 4, 1, 2, 5, 1, 2]      (head MRU ← → LRU tail)

 1: miss [1]
 2: miss [2,1]
 3: miss [3,2,1]
 4: miss full→evict 1  [4,3,2]
 1: miss full→evict 2  [1,4,3]
 2: miss full→evict 3  [2,1,4]
 5: miss full→evict 4  [5,2,1]
 1: hit  move to head  [1,5,2]
 2: hit  move to head  [2,1,5]

hits = 2, misses = 7, final_order = [2, 1, 5]
```

## Constraints

- CPU; `capacity >= 0`; block ids are non-negative integers.
- A hit is determined by "is it currently in the cache", not by the total number of occurrences.
- `final_order` is strictly MRU→LRU and contains only blocks still in the cache (at most `capacity` of them).
