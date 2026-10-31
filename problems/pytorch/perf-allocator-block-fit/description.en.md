# Caching Allocator: Best-Fit Block Allocation, Segment Growth, and Free-Block Coalescing

## Problem

Frameworks do not call `cudaMalloc` per tensor — that costs tens of microseconds and synchronises the device. Instead the framework sublets: it grabs a large segment from the device and carves blocks out of it. PyTorch's caching allocator (`CUDACachingAllocator`) works exactly this way — **block table + best-fit + coalescing on free**. That logic determines the three quantities engineers ask about most:

- **reserved**: how much was actually taken from the device (`torch.cuda.memory_reserved`);
- **fragmentation**: whether freed blocks can be reused in place by later requests, which depends on whether adjacent free blocks were coalesced;
- **segment growth count**: the allocator only asks the device for more when no block fits, and each growth is an expensive large allocation.

This problem simulates the whole ledger in integers: given a stream of `alloc/free` operations, an alignment granularity, and a minimum segment size, output peak reserved bytes, the number of growths, and the final free-block layout.

## Rules

Treat the whole device as one contiguous address space starting at 0. Every request is rounded up to a multiple of `align` to get `need`; the free set always stores **maximal** blocks only (adjacent free blocks must be coalesced immediately) and is sorted by `start`.

For each `alloc`:

1. `need = ceil(nbytes / align) * align`;
2. **best-fit** over blocks with `size >= need`: smallest `size`, ties broken by smaller `start`;
3. carve `need` off the head; if the block is larger, the remainder `[start+need, start+size)` stays free;
4. if nothing fits, **grow first**: `grow = max(need, ceil(grow_unit/align)*align)`, add `grow` to `reserved`, insert the new segment `[old reserved, old reserved+grow)` into the free set (coalescing with neighbours), then **run best-fit again** — it may land on an old small block rather than the new segment;
5. `reserved` never shrinks, and `peak_reserved` is its historical maximum.

For each `free`: return `[start, need)` to the free set and coalesce once with the left neighbour and once with the right neighbour (one release can chain three blocks into one).

Return a six-tuple:

- `peak_reserved`, `final_reserved`: peak/final reserved bytes;
- `num_grow`: number of segment growths;
- `num_free_blocks`: free-block count at the end;
- `largest_free`: largest free block in bytes at the end (0 if none);
- `free_blocks`: final free blocks `[[start, size], ...]` in ascending `start`.

## Example

With `align=512`, `grow_unit=4096`:

```
alloc a 1000 : need=1024, nothing free -> grow 4096 (reserved=4096)
               a=[0,1024)            free=[[1024,3072]]
alloc b 3000 : need=3072, exact fit    b=[1024,3072)   free=[]
alloc c 600  : need=1024, nothing free -> grow to 8192
               c=[4096,1024)         free=[[5120,3072]]
free  b      : no neighbours           free=[[1024,3072],[5120,3072]]
alloc d 2000 : need=2048, two 3072 blocks -> take the smaller address
               d=[1024,2048)         free=[[3072,1024],[5120,3072]]
free  c      : left [3072,1024) and right [5120,3072) -> two coalesces
                                     free=[[3072,5120]]
alloc e 5000 : need=5120, exact fit    e=[3072,5120)   free=[]
alloc f 512  : grow to 12288           f=[8192,512)    free=[[8704,3584]]
free  a      : not adjacent to [8704,3584)  free=[[0,1024],[8704,3584]]
```

```
peak_reserved   = 12288
final_reserved  = 12288
num_grow        = 3
num_free_blocks = 2
largest_free    = 3584
free_blocks     = [[0, 1024], [8704, 3584]]
```

## Constraints

- The op stream is valid: `id` is unique, `free` only targets a currently live block, and `nbytes >= 1` (the third field of `free` is always 0).
- `align >= 1`, `grow_unit >= 1`; neither must be a power of two, but the growth amount must be rounded up.
- Best-fit ties are fixed: smaller `size` first, then smaller `start` — not "first block that fits".
- After growing you must re-run best-fit (not allocate straight from the new segment).
- Fixed return types: the first five fields are `int`, `free_blocks` is `list[list[int]]`.
