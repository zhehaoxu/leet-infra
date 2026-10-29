# Distributed: FSDP Wrap Policy and Per-Block Memory-Communication Accounting

## Description

Plan parameter blocks for a simplified sharded training schedule, then count transient full parameters and collective calls. Use a **toy adjacent-layer packing policy**: accumulate positive layer sizes in order, adding the next layer whenever the total stays at or below `wrap_size`; otherwise start a new block. Layers cannot be split, a single layer cannot exceed the cap, and an exact fit is allowed.

Here `wrap_size` is a **maximum** block size, not PyTorch's recursive, minimum-threshold `size_based_auto_wrap_policy`. This problem does not reproduce its module-tree traversal rules.

## Accounting rules

- Forward visits blocks in order: parameter all-gather → compute → parameter reshard.
- Backward visits blocks in reverse order: parameter all-gather → compute → parameter reshard, plus gradient reduce-scatter.
- Each block contributes **3 collective calls**: forward parameter all-gather, backward parameter all-gather, and gradient reduce-scatter. Resharding only releases full parameters in this model and adds no communication call. These are not network rounds, per-rank counts, or hardware transfer steps.
- `prefetch_distance` **includes the current block**. Before each computation, keep full parameters for the current block and up to `prefetch_distance - 1` consecutive upcoming blocks in the traversal direction. Use the same window in forward and backward, with no assumed forward/backward overlap.
- Peak counts **transient full parameters only**, not bytes or total device memory. Exclude permanent parameter shards, gradients, and activations.

For block sizes \(b_0,\ldots,b_{n-1}\) and \(d=\text{prefetch_distance}\),

\[
\text{peak}=\max_{0\le i<n}\sum_{j=i}^{\min(i+d,n)-1}b_j,
\qquad \text{collective calls}=3n.
\]

Reverse traversal has the same peak over consecutive windows, so forward and backward peaks are not added. An empty layer list has zero blocks, peak, and calls. The starter owns the full input/output types and error contract.

## Examples

```text
param_sizes=[30, 40, 50, 20], wrap_size=80, prefetch_distance=2
block_sizes=[70, 70]
peak_params=140
Return (2, [70, 70], 140, 6)
```

After placing 30 and 40 in the first block, 50 no longer fits. The second block contains 50 and 20. Keeping both blocks resident gives a peak of 140 and six collective calls overall.

```text
param_sizes=[90, 20, 90], wrap_size=100, prefetch_distance=2
Return (3, [90, 20, 90], 110, 9)
```

Windows must be consecutive: the two 90s never share a two-block window, so the peak is 110, not 180. Distance 1 keeps only the current block; a distance larger than the block count keeps at most all blocks.
