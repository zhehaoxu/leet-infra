# ZeRO 1-D Shard Ownership: Splitting Parameters Across Rank Boundaries

## Problem

When ZeRO-1/2/3 partitions the training state along data parallelism, the first step is to **flatten** all parameters into one large 1-D buffer, then cut it into contiguous shards of as equal length as possible, with each rank exclusively owning one shard for optimizer updates. The trouble is that a **parameter** (or its corresponding gradient/optimizer state) occupies a contiguous interval in the flattened buffer, and that interval may exactly **cross the boundary between two shards**—so it is split between two ranks, each updating half. In the DeepSpeed implementation this is the repeatedly troublesome "ownership offset": when deciding ownership, whether the boundary is open or closed, and whether a parameter sitting exactly on a boundary counts as crossing it. One off-by-one mistake and a parameter is missed or miscounted.

This problem gives you a pure integer model: compute how many elements each rank receives, and which parameters cross shard boundaries.

## Sharding Model (follow this definition exactly)

Let the parameters be arranged in a list in flattened order, with the $i$-th parameter having $\texttt{numels}[i]$ elements and the total element count $\texttt{total}=\sum_i \texttt{numels}[i]$. Split it along data-parallel degree $W$ into $W$ contiguous shards of **as equal length as possible**:

- The shard boundaries are $b_r=\left\lfloor \dfrac{\texttt{total}\cdot r}{W}\right\rfloor,\; r=0,1,\dots,W$ (note $b_0=0,\ b_W=\texttt{total}$).
- Rank $r$ ($0\le r<W$) **exclusively owns** the global-index half-open interval $[\,b_r,\ b_{r+1}\,)$, i.e. the number of elements it is responsible for is $b_{r+1}-b_r$.

A parameter $i$ occupies the half-open interval $[s_i,\ e_i)$, where $s_i=\sum_{j<i}\texttt{numels}[j]$ and $e_i=s_i+\texttt{numels}[i]$. If there exists an **internal** boundary $b_r$ ($1\le r\le W-1$) satisfying

$$ s_i < b_r < e_i $$

(strictly inside the parameter interval; coinciding endpoints do not count), then the parameter **crosses** a shard boundary and is shared by two ranks.

## Rules

Implement `solve(param_numels, world)`:

- `param_numels`: a list of non-negative integers giving the number of elements of each parameter in flattened order;
- `world`: data-parallel degree $W\ge 1$.
- Return `(per_rank, straddle)`:
  - `per_rank`: an `int64` tensor of length $W$, where `per_rank[r]` is the number of elements owned by rank $r$ (their sum equals `total`);
  - `straddle`: an ascending `int64` tensor whose elements are the 0-based indices of parameters that **cross a shard boundary**; if none, return an empty tensor.
- When $W=1$ there is only one shard and no internal boundary, so `straddle` must be empty.

## Examples

```
param_numels = [3, 4, 2, 5, 6], world = 4
  total = 20, boundaries b = [0, 5, 10, 15, 20]
  each shard: [0,5) [5,10) [10,15) [15,20)  → per_rank = [5, 5, 5, 5]
  parameter intervals: p0=[0,3) p1=[3,7) p2=[7,9) p3=[9,14) p4=[14,20)
    p1 contains boundary 5 (3<5<7) → straddles
    p3 contains boundary 10 (9<10<14) → straddles
    p4 contains boundary 15 (14<15<20) → straddles
  straddle = [1, 3, 4]
→ (tensor([5,5,5,5]), tensor([1,3,4]))
```

## Constraints

- `0 ≤ len(param_numels) ≤ 1024`, each `numels[i] ≥ 0`; `1 ≤ world ≤ 1024`
- Return two `int64` tensors `(per_rank, straddle)`; compared with `torch.equal`
- This is a pure integer ownership model; it does not involve padding / communication / real GPU memory
