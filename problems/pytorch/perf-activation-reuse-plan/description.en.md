# Activation Memory Reuse: Offset Planning over Lifetime Intervals with a Peak Lower Bound

## Problem

In a training or inference graph, an activation tensor's **lifetime does not span the whole graph**: once its last consumer has run, the buffer can be released and a later tensor can reuse that memory. Compilers and frameworks therefore run an **offline memory plan**: give every tensor an address offset, requiring only that two tensors whose **lifetimes overlap** never share bytes; tensors with disjoint lifetimes may be stacked at the same address. TensorFlow's grappler memory optimizer and TorchScript's static memory pool both do this.

Plan quality is measured by the **peak**: the maximum `offset + size` over all tensors. A natural lower bound is the maximum, over any point in time, of the byte sum of **simultaneously live** tensors — but it is not always attainable (with unequal sizes, gaps may not fit later arrivals), which is why this problem asks for both numbers.

## Rules

Given tensors with lifetime `[start, end)` (logical steps, half-open) and request bytes `nbytes`, plus alignment `align`. Two tensors **overlap in lifetime** iff `max(s1, s2) < min(e1, e2)`.

1. Actual footprint `size = ceil(nbytes / align) * align`; offsets must also be multiples of `align`;
2. Placement order: `(size descending, start ascending, end ascending, original index ascending)`;
3. For each tensor in that order, pick the **smallest** feasible offset `o`: `[o, o+size)` must not intersect the byte ranges of any **already placed tensor whose lifetime overlaps** (already placed tensors with disjoint lifetimes impose no constraint);
4. Peak `peak = max(o + size)`;
5. Lower bound `lower_bound = max over t (Σ size of tensors live at t)`, with `t` ranging over all `start` values.

Return `(offsets, peak, lower_bound)`; `offsets` follow the **input order**.

## Example

`align=1024`, five tensors (sizes already aligned):

```
i0 [0,4) 3000 -> 3072
i1 [1,3) 2000 -> 2048
i2 [2,6) 5000 -> 5120
i3 [4,6) 2000 -> 2048
i4 [5,8) 1000 -> 1024

placement order (size descending): i2, i0, i1, i3, i4
  i2 first: off 0                                  [0,5120)
  i0 [0,4) overlaps i2 [2,6) -> 0 taken -> off 5120    [5120,8192)
  i1 [1,3) hits i2, i0       -> off 8192               [8192,10240)
  i3 [4,6) only hits i2      -> off 5120 works         [5120,7168)
```

`i4`'s candidate offsets are "0 plus the ends of already placed lifetime-overlapping tensors" = {0, 5120, 7168}: 0 hits i2 and 5120 hits i3, while **7168 works**, giving `[7168,8192)`.

```
offsets     = [5120, 8192, 0, 5120, 7168]
peak        = 10240      # from i1: 8192 + 2048
lower_bound = 10240      # at t=2: i0+i1+i2 = 3072+2048+5120
```

## Constraints

- `end > start`, `nbytes >= 1`, `align >= 1`; empty input is allowed (empty list, peak 0, bound 0).
- Offsets must be the **smallest feasible** ones, not merely any feasible ones: the tests shift each tensor left by one slot and assert it then collides with some earlier-placed, lifetime-overlapping tensor.
- `peak >= lower_bound` always holds, but **equality is not required**: with unequal sizes, greedy-by-size leaves gaps (see the fragmentation case in the tests).
- Fixed return types: `list[int]`, `int`, `int`.
