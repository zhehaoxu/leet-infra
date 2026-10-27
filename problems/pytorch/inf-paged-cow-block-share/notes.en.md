# Solution Notes · Page Tables and Copy-on-Write

## Keep the two index spaces apart

The easiest thing to get wrong in `append` is which pages actually get written. Derive it from the token range:

```
first = old_tokens // block_size
last  = (old_tokens + arg - 1) // block_size
```

then walk `first..last`. Deciding by "how many pages the sequence has" or by "does it share anything" produces the two classic bugs: copying every page when none needed it, or missing the part that spills onto a new page.

## Compute the page bounds from the pre-append token count

`first` uses the **old** token count. When it is an exact multiple of `block_size`, `first` points at the page *after* the last existing one, so `first >= len(table)` and the loop takes the allocation branch rather than the copy branch. That is exactly the "an exactly-full page is not written" rule.

## Sharing has exactly one test: refcount > 1

A block is copied because *I am about to write it*; whether someone else also holds it is answered only by the refcount. Do not keep a per-sequence `is_shared` flag — it has to be kept in sync across fork and free and drifts out of agreement with the counter. The refcount is the single source of truth.

## Fork is pure bookkeeping

After a fork the two page tables have identical contents but are two separate lists; a later append on one side only touches its own list. The shallow copy `tables[dst] = list(tables[src])` is therefore mandatory — share the list object and both the page tables and the copy count go wrong.

## The final snapshot must include freed sequences

`free` removes a sequence from the live table, but the output has one row per sequence id, so released sequences still get a row. Keep the live table and the final-snapshot table separate: append mutates the same list object so the snapshot follows automatically, while free only deletes the live entry.

## Complexity

Each op is O(number of touched pages); allocating a free block is a linear scan over `num_blocks`, which a free list turns into O(1). Real paged attention runs tens of thousands of blocks, so the scan is only fine at teaching scale.

## Related problems

- `inf-paged-kv-block-alloc`: the same block pool, but only sequential admission — no indirection and no sharing.
- `inf-kv-block-admission`: who gets to run when the block budget is short.
- `inf-lru-kv-cache-eviction`: the reclaim side of the same pool.
