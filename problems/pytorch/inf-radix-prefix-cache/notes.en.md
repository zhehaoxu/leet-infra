# Solution Notes · Radix-Tree Prefix Cache

## Origin

- SGLang: *Efficient Execution of Structured Language Model Programs* (arXiv:2312.07104, NeurIPS 2024) — introduces RadixAttention: a token-level compressed prefix tree over KV, with per-node reference counts, edge splitting on a partial match, and leaf-wise LRU eviction under memory pressure.
- vLLM's automatic prefix caching does the same at block-hash granularity; this problem deliberately works at token granularity so it stays distinct from the block-hit accounting in `inf-prefix-cache-block-hit`.
- The edge-compressed trie (radix / Patricia tree) is the classical structure behind it: chains with a single child are merged into one edge, so the node count scales with "branch points plus request ends" rather than with prompt length.

## Key points and pitfalls

**The hit is a token count, not a node count.** A request may match only the first half of an edge (in Example 1, `[10,11,99]` hits 2). The edge must be split at the match point, otherwise the later `[10,11,12]` would be counted as a full hit or a full miss.

**A split adds a node; it does not shorten the parent.** The original node keeps the unmatched tail and hangs under the new middle node. The cached token total is unchanged (same tokens before and after), only the cut becomes finer.

**An active request pins its whole path.** Eviction candidates must exclude leaves whose path is a prefix of some active request's prompt. Example 3 keeps 6 tokens under `capacity = 4` because both leaves are in use. Expressing reference counting as "active set + prefix test" is much harder to get wrong than moving counters around during splits.

**`finish` does not refresh last-used.** It only releases the claim; refreshing here would scramble the LRU order of Example 2. Hits, on the other hand, must refresh — every node visited on the way down — or a just-reused prefix would be treated as cold data and dropped first.

**Only leaves are evicted, and the whole edge goes with them.** After a drop the parent may become a new leaf, so candidates are recomputed within the same operation; that is what keeps shared prefixes (e.g. `[1,2]` after its child edge `[3,4]` is dropped) alive for future hits.

**Ties break on node id.** Leaves touched or created at the same instant are ordered by creation id, so the result stays unique.

**The ledger closes.** `cached_tokens == Σ prompt length − Σ hits − evicted_tokens`; this is the strongest cross-check and it immediately exposes any missed insertion or eviction.

## Verification

- Reference: a node tree with parent pointers (each node stores `edge/path/children/parent/last/id`), incrementally maintaining `cached` and `evicted`; eviction collects candidates with one DFS and removes the minimum.
- Independent oracle: builds **no tree** at all — it keeps a flat map "full token path → (last used, id)", recovers parentage from the longest proper prefix, defines a leaf as "not a prefix of any other stored path", and evicts by deleting one entry and reclaiming `len(path) − len(parent(path))` tokens. The two representations share no data structure.
- 3000 random operation streams (tokens 0–3, prompt length 1–6, capacity 1–12, random finish timing) agree bit-for-bit; seven hand-computed scenarios (split, LRU, hit refresh, pinned overflow, surviving parent path, no shared prefix, unfinished request pinning) plus the random invariant `cached == inserted − evicted` back them up.

## Complexity

- A single request walks at most `O(L)` tokens (`L` = prompt length) and inserts `O(L)`.
- Each eviction does one DFS over all nodes, `O(nodes)`; since every round removes at least one node, one operation costs `O(nodes × removed)`.
- Overall bound `O(Σ L_i × N)` with `N ≤ Σ L_i ≤ 64 × 16 = 1024` nodes — milliseconds. Production engines get amortised `O(L)` per step with hash maps and intrusive lists.

## Related problems

- `inf-prefix-cache-block-hit`: block-hash prefix hits — whole-block granularity, no tree and no eviction.
- `inf-lru-kv-cache-eviction`: block-level LRU eviction; this problem is token-level and demands tree reuse.
- `inf-paged-kv-block-alloc`: physical block allocation and page tables — the other half of the same cache ledger.
