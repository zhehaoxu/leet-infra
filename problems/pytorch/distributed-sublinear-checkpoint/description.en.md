# Sublinear Activation Checkpointing: Peak Memory of the √n Recomputation Schedule

## Problem

Backpropagation needs the activations computed in the forward pass. The most time-efficient approach is to **store everything** ($n$ layers occupy $n$ copies of activation memory); the most memory-efficient is to **store nothing** and recompute the entire forward pass from scratch during backward. Gradient checkpointing (activation checkpointing) takes the middle road: **store one checkpoint every $k$ layers**, and when backward reaches a segment, **recompute** the intermediate activations starting from the checkpoint at that segment's start.

Chen et al. (2016, *Training Deep Nets with Sublinear Memory Cost*) proved that with a sensible choice of checkpoints, the activation memory of an $n$-layer network can be reduced to $O(\sqrt n)$, at the cost of only about $\sqrt n$ additional forward passes. This problem quantifies that result as a pure integer optimization: given the number of layers and the activation bytes per layer, find the checkpoint interval $k$ that **minimizes peak memory**, and report the peak and the store-all baseline.

## Peak Model (follow this definition exactly)

View the network as $n$ **sequential** layers, each forward pass producing one copy of activation occupying $a=\texttt{act\_bytes}$ bytes. With the schedule "store a checkpoint every $k$ layers" ($1\le k\le n$), the **peak activation memory** during the backward traversal is:

$$ \text{peak}(k) = \Big(\underbrace{\lceil n/k\rceil}_{\text{resident checkpoints}} + \underbrace{k}_{\text{max activations alive at once in the current segment's recompute window}}\Big)\cdot a $$

- First term: checkpoints are placed along the whole chain at interval $k$, so the number of resident checkpoints is bounded by $\lceil n/k\rceil$;
- Second term: on entering a segment during backward, recomputation starts from that segment's start, and at most $k$ activations are alive simultaneously within the segment.

The function $\lceil n/k\rceil + k$ attains its minimum $\approx 2\sqrt n$ at $k\approx\sqrt n$, which is exactly the source of the sublinear $O(\sqrt n)$ memory.

Also give the store-all baseline (no checkpointing, every layer stored):

$$ \text{store\_all} = n\cdot a $$

## Rules

Implement `solve(n_layers, act_bytes)` and return the triple `(best_k, min_peak, store_all)`:

- `best_k`: the $k\in[1,n]$ that minimizes $\text{peak}(k)$; **if several $k$ tie for the minimum, take the smallest $k$**;
- `min_peak`: $\text{peak}(\texttt{best\_k})$ (bytes, integer);
- `store_all`: $n\cdot a$ (bytes, integer).
- All three are integers (Python `int`; do not use floating point, which loses precision).

## Examples

```
n_layers=9, act_bytes=1
  peak(k) = ceil(9/k)+k:  k=1→10, k=2→7, k=3→6, k=4→7, ... k=9→10
  minimum at k=3, peak=6
  store_all = 9
→ (3, 6, 9)          # √9=3, peak 6≈2√9, store-all needs 9

n_layers=100, act_bytes=1_000_000
  best_k=10, peak=(ceil(100/10)+10)*1e6=(10+10)*1e6=2e7, store_all=1e8
→ (10, 20000000, 100000000)
```

## Constraints

- `1 ≤ n_layers ≤ 100000`, `1 ≤ act_bytes ≤ 1e9` (integers)
- Return `(int, int, int)`; equality is checked with exact integer comparison
- Only enumeration/optimization of the peak model above is performed; the recomputation graphs and memory allocators of real frameworks are not involved
