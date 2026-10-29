## Deriving groups from strides

Accumulate axis sizes along `order` to obtain strides. A rank's coordinate on one axis is `(rank // stride) % size`. Subtract that coordinate's contribution to obtain the group's minimum member `base`; all members are `base + k*stride`, already sorted.

The local rank equals the coordinate. A group ID removes this axis's digit from the mixed-radix rank:

$$group\_id=rank\bmod stride+\lfloor rank/(stride\cdot size)\rfloor\cdot stride.$$

Lower digits remain unchanged and higher digits compact by one position. The resulting IDs are dense from zero and match ordering groups by their minimum member.

## Layout pitfalls

`order` changes rank interpretation, not just display order. Tests keep dictionary insertion order fixed and enumerate all 24 four-axis permutations, catching solutions that ignore `order`. A size-one axis produces singleton groups with local rank zero; an axis spanning the whole world produces one group.

First-axis-fastest here differs from conventional last-axis-fastest row-major reshape. A device-mesh dimension declaration must respect the chosen encoding rather than blindly applying reshape.

## Independent verification and complexity

The oracle enumerates the Cartesian product with reversed axes, assigns rank IDs in sequence, then filters coordinates equal on every other axis. It does not reuse the stride formula. Additional checks verify each axis partitions the world exactly, members agree on groups and IDs, and local ranks index back to their owners.

With $A$ axes and total axis size $D$, explicitly returning every group for every rank requires $O(WD)$ time and output space. Strides and a current coordinate require $O(A)$ auxiliary storage. The output itself imposes this cost.
