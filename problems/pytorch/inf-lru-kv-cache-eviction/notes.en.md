# Solution Notes · LRU Eviction

## O(1) implementation with a doubly linked list + hash map

Production-grade LRU uses a `HashMap<key, Node>` paired with a doubly linked list: on a hit, unlink the node and insert it at the head in `O(1)`; if not full, insert at the head directly; when full, the tail node is the LRU and can be deleted in `O(1)`. The scale here is small, so `list`'s `in / remove / insert(0)` suffice; the oracle uses `OrderedDict` (`move_to_end` + `popitem(last=False)`), which is a different implementation path, and both must agree.

## The pitfall in hit semantics

A "hit" means **currently still in the cache**, not "this id has appeared before". Accessing an id after it has been evicted = miss (it must be reloaded). This is exactly why a preempted prefix cache has to recompute prefill.

## Difference from real KV eviction

In real systems the eviction unit is a **block** and there are reference counts: a block currently used by an active request cannot be evicted (`refcount>0`), and only blocks in the free queue are LRU candidates; a distinction is also drawn between "evictable idle KV" and "hot KV still decoding". This problem treats each access as a touch on some block, abstracts away reference counting, and focuses on the pure LRU temporal logic.

## capacity boundaries

When `capacity==0` the cache is always empty and no access can reside (everything misses), so `final_order==[]`; `capacity==1` degenerates to "only an immediately repeated access hits". These two boundaries are the easiest to get wrong, and tests cover them specifically.
