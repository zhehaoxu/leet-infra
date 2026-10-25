# Explained: ZeRO 1-D Shard Ownership

## Source
DeepSpeed ZeRO (Rajbhandari et al. 2019, arXiv:1912.05771). ZeRO **flattens parameters/gradients/optimizer states into a 1-D contiguous buffer**, then cuts it into `world` shards of as equal length as possible along the data-parallel dimension, with each rank maintaining only its own shard. The real implementation pads to handle non-divisible cases; this problem captures the same thing with a `floor(total*r/world)` boundary model. The core question is the ownership decision "a single parameter that crosses a shard boundary is split and assigned to two ranks."

## Core Points and Common Mistakes
1. **Boundaries are half-open intervals**: rank $r$ owns $[b_r, b_{r+1})$, $b_r=\lfloor total\cdot r/W\rfloor$. Parameter $i$ owns the interval $[s_i,e_i)$, and the straddle criterion is "there exists a boundary that lies **strictly inside**" $s_i<b_r<e_i$. **Coinciding endpoints do not count**—a parameter that exactly fills one shard (e.g. `[6,6,6,6]` with $W=4$) does not straddle; this is the most typical off-by-one.
2. **One parameter can cross multiple boundaries**: a single large parameter (e.g. `[5]` with $W=3$, $b=[0,1,3,5]$) contains boundaries 1 and 3 internally and is shared by 3 ranks, but it counts as only **one** straddle index—deduplicated and recorded by index, not counted per boundary.
3. **`per_rank` always sums to total**: the `floor` boundaries guarantee shard lengths differ by at most 1, with `per_rank[r]=b_{r+1}-b_r`.
4. **Zero-element parameters**: when `numel==0` the interval is empty and it never straddles.
5. **$W=1$ degeneracy**: only $b_0=0,b_1=total$, with no internal boundary, so `straddle` must be empty.

## Cross-Checking Criteria
Reference solution: compute the boundary array first, then for each parameter test straddling with the "strictly inside the interval" inequality $s<b_r<e$. The judge's `_oracle` takes a **structurally different** path: it assigns every global index $x$ to an owner via interval-membership scanning; `per_rank` comes from counting, and straddling from "more than one owner appears in the set of indices covered by the parameter." The two paths cross-validate the same boundary model. There are also three groups of assertions: `world=1` degeneracy, parameters exactly aligned with boundaries not straddling, and a single parameter crossing multiple boundaries.

## Complexity
Let $P$ be the number of parameters: \(O(P\cdot W)\) (scanning boundaries per parameter); `_oracle` is \(O(total\cdot W)\) index scanning.
