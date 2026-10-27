# Solution Notes · multi-LoRA Adapter Residency

## Two sets and one queue

Three roles live in this state machine, and confusing any two breaks it:

- **resident (set)**: what is currently in memory, used to answer "hit". Contains pinned and non-pinned.
- **lru (ordered list)**: **only the non-pinned resident adapters**, head = least recently used. Pinned adapters never enter it.
- **pinned (set)**: resident from the start and never a swap candidate.

A hit is `a in resident`; whether a load is possible is `lru non-empty`. Keep the two tests apart.

## A hit touches the queue — except for pinned

Hitting a pinned adapter refreshes nothing (it is not in `lru`). Hitting a non-pinned one must remove it from `lru` and append it at the tail — without that refresh the policy is FIFO and `1,2,1,3` evicts the wrong adapter.

## Evict before loading

The order is "free a slot, then load", not "load, then free". Loading first briefly exceeds `capacity`, which inflates `peak_bytes` by one adapter, and rollback-on-overflow makes the logic needlessly convoluted.

## Rejection is not "do nothing"

A rejected request indeed changes no state, but `evicted` must still get a `-1` and `hit` a `False`, positionally aligned with the request sequence. Skipping the write shifts every later index.

## The initial `peak_bytes`

The peak starts at the pinned byte sum, not 0: pinned adapters are already resident before step 0. Miss this and it only shows up when `pinned` is non-empty and no swap-in ever happens — exactly the combination that is easiest to forget to test.

## How it differs from the KV-side LRU

`inf-lru-kv-cache-eviction` manages KV blocks: capacity counted in blocks, no pinning. This one manages adapters: capacity counted in adapters, pinning, and memory accounted in bytes rather than count. Same skeleton, but those two differences make a copy-paste implementation wholly wrong.

## Complexity

Worst case O(capacity) per step (finding the LRU is O(1), but summing bytes is O(capacity)). Production code keeps a running byte total so each step is O(1).

## Related problems

- `inf-lru-kv-cache-eviction`: the same eviction idea over KV blocks, without pinning.
- `peft-lora-memory`: the training-side adapter parameter/memory formula, complementary to this runtime ledger.
- `inf-cache-aware-routing`: adapter affinity is another term a router can price in.
