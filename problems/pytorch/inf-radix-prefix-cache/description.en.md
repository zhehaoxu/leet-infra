# Radix-Tree Prefix Cache: Longest-Prefix Hits and LRU Eviction

## Problem

Multi-turn chat, few-shot prompts and agent system prompts make a large fraction of requests share the same prompt prefix. Keeping the KV cache of that prefix lets the next request skip the matching prefill — that is prefix caching. Production engines (SGLang RadixAttention, vLLM automatic prefix caching) organise the cache as a **token-level, edge-compressed trie**: every node holds one contiguous run of tokens, nodes themselves own no capacity, and the cache size is the sum of all edge lengths.

Your job is the core of that tree: walk the longest common prefix for an arriving request, insert the part that missed, keep the KV of finished requests around for later reuse, and, when the total exceeds capacity, drop the whole edge of a least-recently-used leaf that nobody is using.

## Rules

`ops` is a time-ordered list of operations; the timestamp of an operation is its index. Each element is one of:

- `("request", rid, tokens)`: request `rid` arrives with prompt `tokens` (an integer token sequence, length `>= 1`);
- `("finish", rid)`: request `rid` ends and releases its claim on the cache.

**Handling a request**:

1. Descend from the root along the longest common prefix with `tokens`. If an edge matches only its first part, split that edge at the match point: the new middle node owns the matched part, the original node keeps the rest and becomes its child. The same split applies when `tokens` ends in the middle of an edge.
2. Every node visited on the way down (including the freshly split node) has its "last used" timestamp refreshed to the current time.
3. The number of matched tokens is `hit`. If tokens remain, the whole remainder becomes one new edge under the match point (its last-used timestamp is also the current time); if nothing remains, no node is created.
4. The request joins the **active set** and claims the KV along its whole `tokens` path until it is finished.

**Handling a finish**: the request leaves the active set. Its KV stays in the cache for later requests; the nodes it pinned become evictable from this step on. `finish` itself does not refresh any last-used timestamp.

**Capacity**: after every operation, while the total cached token count exceeds `capacity`, evict repeatedly:

- candidates are nodes that are **leaves** and whose path is **not a prefix of any active request's prompt**;
- pick the candidate with the lexicographically smallest `(last used, node id)` and delete its whole edge together with the node;
- if a new leaf appears as a result, recompute the candidates in the next round.

If no candidate exists, stop — while active requests fill the tree, the cache is allowed to exceed `capacity` temporarily.

**Node ids**: the root is 0; all other nodes are numbered from 1 in creation order (a split-created node counts as a creation).

Return a triple:

- `hits`: per `request` in arrival order, the number of matched tokens;
- `evicted_tokens`: total tokens evicted over the whole run;
- `cached_tokens`: tokens left in the cache after the last operation.

## Examples

**Example 1** (longest prefix plus a mid-edge split): `capacity = 32`

```
t=0  request 0 [10,11,12,13]   empty cache: hit 0, one edge (10,11,12,13)
t=1  finish 0                  the KV stays cached
t=2  request 1 [10,11,99]      hit 2; edge (10,11,12,13) splits into (10,11) and (12,13)
                               the remaining (99) hangs under (10,11)
t=3  finish 1
t=4  request 2 [10,11,12]      hit 3
t=5  finish 2
t=6  request 3 [10]            hit 1
t=7  finish 3
```

```
hits           = [0, 2, 3, 1]
evicted_tokens = 0
cached_tokens  = 5          # (10,11) + (12,13) + (99)
```

**Example 2** (LRU drops the coldest leaf): `capacity = 6`

```
t=0  request 0 [1,2,3]     cached 3
t=1  finish 0
t=2  request 1 [4,5,6]     cached 6
t=3  finish 1
t=4  request 2 [7,8,9]     cached 9 > 6; leaves (1,2,3) at t=0, (4,5,6) at t=2,
                           (7,8,9) at t=4 but pinned by an active request -> evict (1,2,3)
t=5  finish 2              cached 6
t=6  request 3 [4,5,6]     hit 3
t=7  finish 3
```

```
hits           = [0, 0, 0, 3]
evicted_tokens = 3
cached_tokens  = 6
```

**Example 3** (active requests push the cache over capacity): `capacity = 4`

```
t=0  request 0 [1,2,3,4]   cached 4
t=1  request 1 [5,6]       cached 6 > 4, but both leaves are pinned by active requests -> evict nothing
t=2  finish 0              (1,2,3,4) becomes evictable, its whole edge is dropped, cached 2
```

```
hits           = [0, 0]
evicted_tokens = 4
cached_tokens  = 2
```

## Constraints

- `capacity >= 1`; `ops` is non-empty and holds at most 64 `request` operations; prompt length `>= 1`.
- `rid` is an integer; each `rid` is requested once, finished at most once, and a `finish` only targets a currently active request.
- Tokens are arbitrary integers; only equality matters.
- `hits` is a Python `int` list and the other two values are Python `int`s. Only the numbers are judged; any implementation style is fine (parent-pointer node tree, path set, recursion).
