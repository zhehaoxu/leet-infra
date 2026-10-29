# Distributed: Parallel-Mesh Rank Mapping and Communication Groups

## Scenario

Arrange $W$ ranks along independent axes such as TP, CP, DP and PP. An axis communication group varies only that axis's coordinate while keeping all others fixed. Produce coordinates, group members, local ranks, and a separately numbered group ID for every axis.

## Least-significant-axis-first encoding

The supplied `order` defines the layout: **earlier axes change faster**, independently of dictionary insertion order. Axis $j$ has size $d_j$ and stride $s_j=\prod_{k<j}d_k$:

$$rank=\sum_j c_j s_j,\qquad c_j=\lfloor rank/s_j\rfloor\bmod d_j.$$

For example, `['tp','cp','dp','pp']` means

$$rank=c_{tp}+tp\,c_{cp}+tp\,cp\,c_{dp}+tp\,cp\,dp\,c_{pp}.$$

This is **least-significant-axis-first** mixed-radix encoding. Conventional row-major tensors have the last axis changing fastest; reshaping with the same axis order is incorrect. Reverse the axis order when constructing an equivalent row-major mesh.

## Group ownership

For each axis:

1. Fix every other coordinate, enumerate this axis, and sort the resulting member ranks.
2. The local rank is the current rank's index in that member list, equal to its coordinate on this axis.
3. Order distinct groups by their minimum member and assign dense IDs $0,1,\ldots,W/d_j-1$. Group IDs belong to one axis only, not a shared cross-axis namespace.

The mesh must cover all ranks, and the order must contain each axis exactly once.

## Example

```text
world_size=8, dims={'tp':2,'dp':4}, order=['tp','dp']
rank 0: coord={'tp':0,'dp':0}
        groups={'tp':[0,1], 'dp':[0,2,4,6]}
        local_ranks={'tp':0,'dp':0}, group_ids={'tp':0,'dp':0}
rank 1: coord={'tp':1,'dp':0}
        groups={'tp':[0,1], 'dp':[1,3,5,7]}
        local_ranks={'tp':1,'dp':0}, group_ids={'tp':0,'dp':1}
```

This is a pure integer simulation, not real process-group creation. Non-independent layouts such as EP and DP sharing a rank dimension are outside scope.
