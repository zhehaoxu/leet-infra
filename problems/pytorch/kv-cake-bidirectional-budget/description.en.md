# KV Cache: CAKE Layer-Preference Budget and Cascading Propagation

## Problem

The naive cross-layer budget is a "pyramid"—shallow layers keep more, deep layers keep less, one fixed formula everywhere. In practice layer preferences vary enormously across models and tasks: in some models the middle layers carry the retrieval load while deep layers need very little cache. CAKE (Qin et al., ICLR 2025, arXiv:2503.12491) frames this as a "**cake-slicing problem**": measure each layer's **preference**, slice the global budget according to it, and propagate information across layers in a **cascading** manner.

This problem keeps two deterministic cores of that idea.

**(1) Preference and budget apportionment.** Let layer \(l\) have the accumulated importance row \(s_l\in\mathbb R^{L}\) and let its preference be the total mass \(m_l=\sum_j s_l[j]\); the global budget is \(B\) (how many positions the whole model may keep). Apportion by mass share:

\[
b_l=\left\lfloor \frac{B\,m_l}{\sum_{l'}m_{l'}}\right\rfloor,\qquad
\text{the leftover } B-\sum_l b_l \text{ is handed out one at a time by descending fractional part}
\]

The fractional part is \(\frac{B m_l}{\sum m}-b_l\), with **ties going to the smaller layer index**; a layer never exceeds \(L\). A layer with zero mass receives 0 positions. Layers with more mass get a larger share—that is the adaptive part, replacing the fixed pyramid.

**(2) Cascading across layers.** Layers depend on each other (entities localized by a shallow layer must also be kept deeper down). Given a sequence of dependency edges `edges = [(src, dst), ...]`, propagate importance in the **given order**:

\[
w_{dst}\leftarrow w_{dst}+w_{src}
\]

with \(w_l=s_l\) initially. Because the accumulation follows the edge order, propagation **cascades**: an earlier edge feeds its already-accumulated result forward (`(0,1),(1,2)` means layer 2 sees the evidence of both 0 and 1). After propagation, every layer takes the top-\(b_l\) of its own \(w_l\) (ties to the smaller index) as its final retained set.

This is precisely where it diverges from top-down greedy layer filling: apportionment is a **single global pass** (all layers are sliced at once by preference), while dependencies are **backfilled edge by edge**, so the two steps stay separate and do not interfere.

## Task

Implement `solve(scores, budget, edges)`:

- Input: `scores` (float32 `(Lc, L)`, non-negative); `budget` (non-negative integer \(B\)); `edges` (list of `(src, dst)` with layer indices in `[0, Lc)`; self-loops are invalid).
- Output `(alloc, keep)`: `alloc` is `(Lc,)` int64 (positions per layer); `keep` is `(Lc, L)` bool.
- The leftover is handed out by descending fractional part, ties to the smaller index, looping until it is exhausted or every layer hits `L`; when `B` exceeds the total capacity `Lc·L` every layer caps at `L` and `sum(alloc) < B` is expected.
- Raise `ValueError` for invalid arguments (negative `budget`, negative scores, self-loop, out-of-range layer index).

## Example

```
scores = [[9, 8, 7, 0, 0, 0],      layer 0: the front half matters
          [0, 0, 0, 9, 8, 7],      layer 1: the back half matters
          [1, 2, 3, 4, 5, 6]]      layer 2: later positions matter more
budget = 9, edges = [(0, 1)]

mass = [24, 24, 21] (total 69) -> 9*24/69 = 3.13 -> 3, 3, floor(2.74) = 2
leftover 1 goes to layer 2 (largest fraction 0.74) -> alloc = [3, 3, 3]

w1 = [0,0,0,9,8,7] + [9,8,7,0,0,0] = [9, 8, 7, 9, 8, 7]
layer0 top3 -> {0,1,2}        keep[0] = [T,T,T,F,F,F]
layer1 top3 -> 9(0), 9(3), 8(1, tie -> smaller)   keep[1] = [T,T,F,T,F,F]
layer2 top3 -> {5,4,3}        keep[2] = [F,F,F,T,T,T]
```

## Constraints

- Propagation **writes dst only, never src**: edge order changes the result (`(0,1)` and `(1,0)` differ), so accumulate strictly in the given order.
- Propagation **adds** rather than overwrites: when several edges point at one layer the evidence stacks; a chained edge forwards an already-accumulated upstream, so double counting is intended behaviour, not a bug.
- Apportionment must `floor` first and then distribute the leftover; `round` breaks the sum-equals-budget invariant and depends on floating-point rounding at the boundary.
- A zero-mass layer gets 0 and retains nothing; `budget = 0` yields empty masks everywhere.
