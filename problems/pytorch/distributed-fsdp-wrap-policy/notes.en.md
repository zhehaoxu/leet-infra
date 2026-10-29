## Deriving the partition and its constraints

Every layer size is positive and fits within `wrap_size`. Starting at the first unprocessed layer, choose the longest consecutive prefix whose total fits the cap. If the next layer fits, the current block must continue; otherwise a new block must begin. This defines the ordered greedy partition, not an optimization over arbitrary layer permutations or splits. An exact fit remains valid.

The starter requires a positive integer cap and prefetch distance. Nonpositive layers, oversized layers, and invalid types raise `ValueError`; bool is not an integer for this contract. An empty list is valid, but the cap and prefetch distance must still be validated first.

## Forward and backward event ledger

For each block, the logical events across one training step are:

| Phase | Event order | Collective calls |
| --- | --- | --- |
| Forward | Parameter all-gather → forward compute → parameter reshard | One parameter all-gather |
| Backward | Parameter all-gather → backward compute → parameter reshard; gradient reduce-scatter | One parameter all-gather plus one gradient reduce-scatter |

Forward has already released the full parameters, so backward must all-gather them again. Prefetching may move an all-gather earlier but does not change the total of one gather per block per phase. Here resharding means releasing full parameters while retaining the existing shard; it is not another collective. Gradient reduce-scatter is counted separately.

Thus \(n\) blocks produce \(n+n+n=3n\) **logical collective calls**. The four-tuple retains the legacy field name `comm_rounds`, but it does not count network rounds, calls summed over all ranks, or NCCL hardware steps.

## Deriving the peak

Let \(d\) include the current block. Before forward computation of block \(i\), the resident full parameters belong to blocks \(i\) through \(\min(i+d,n)-1\). Release the current block after computation and gather only newly entering blocks for the next window. The transient peak is the largest such consecutive interval sum, not the sum of the largest \(d\) blocks anywhere in the sequence.

Backward reverses the traversal while retaining the same \(d\). When \(d\le n\), both traversals cover all length-\(d\) consecutive intervals. Because sizes are positive, shorter boundary windows cannot exceed a full window containing them. When \(d>n\), the peak is the total parameter count. The two phases therefore have the same peak; without forward/backward overlap, take their maximum rather than their sum.

For `[90, 20, 90]` and \(d=2\), only neighboring pairs `90+20` and `20+90` can coexist, giving 110. For \(d=1\), the peak is the largest single block; for empty input, it is defined as zero.

## Complexity and independent verification

Let \(L\) be the layer count and \(n\) the block count. The reference partition takes \(O(L)\) time, and directly summing the windows takes \(O(n\min(d,n))\). Total time is \(O(L+n\min(d,n))\), with \(O(n)\) space including the block list and temporary slices. Prefix sums or a sliding window can reduce window accounting to linear time without changing the rules.

The tests do not duplicate the reference greedy loop. Small cases enumerate all feasible consecutive partitions and select the lexicographically largest endpoint sequence, which chooses each longest prefix. They then simulate forward and backward resident sets and all-gather, compute, reshard, and reduce-scatter events to derive peaks and call counts independently.

## Simplified model versus real systems

This is not PyTorch's recursive, minimum-threshold auto-wrap policy. It does not handle module trees, shared parameters, or actual communication. The peak counts transient full parameters only, excluding permanent shards, gradients, activations, communication buffers, and allocator overhead; it is not total device memory. Real prefetch scheduling, collective fusion, sharding strategies, and computation/communication overlap are outside this model.
