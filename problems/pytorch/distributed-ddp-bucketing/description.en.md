# Distributed: DDP Static Gradient Buckets and Readiness

## Description

DDP groups parameter gradients into buckets to amortize collective overhead. This problem is an **educational static size-cap model**, not a reproduction of PyTorch's bucket builder or communication scheduler.

Parameter registration order does not determine actual gradient completion order. Here, reverse parameter-index order is an explicitly chosen **packing order only**; the supplied per-rank readiness times determine when gradients become available. The starter contains the full input/output contract.

## Static packing

Visit parameter indices from last to first. Keep appending while the total size fits the cap; equality fits. If adding the next gradient would exceed the cap, seal the current bucket and begin the next. An individually oversized gradient forms a singleton after sealing any preceding partial bucket. Never split a gradient or sort parameters by their readiness times.

Despite its name, `bucket_cap_mb` is measured in **MiB**, with capacity in bytes equal to `int(bucket_cap_mb * 1024**2)`. Input sizes are positive integer bytes; the converted cap is positive. Empty parameter lists are allowed.

## Readiness and common bucket order

Let \(B_k\) be bucket \(k\), and \(t_{r,i}\) be the given readiness time of parameter \(i\) on rank \(r\):

\[
R_{r,k}=\max_{i\in B_k}t_{r,i},\qquad
G_k=\max_r R_{r,k},\qquad
E_k=\max_{0\le j\le k}G_j.
\]

- \(R\): each rank's bucket is ready only after every member gradient is ready.
- \(G\): the bucket is ready on all ranks, including the slowest one.
- \(E\): its **global synchronized eligibility** respects the common bucket order. An already-ready later bucket cannot overtake an earlier bucket. Equal eligibility times are allowed.

These are not actual local enqueue times or network completion times. No communication durations are given or inferred. With no parameters, each rank has an empty readiness row, and all other result lists are empty.

## No root or start-owner rank

Every rank participates in every bucket's all-reduce and receives the reduced result. All-reduce has no designated root/start-owner rank: even a ring algorithm involves all ranks, not bucket \(k\) being owned by rank \(k\bmod W\).

Real DDP can rebuild buckets, group by dtype/device, and synchronize asynchronously while backward computation continues. A bucket fitting its byte cap alone does not imply that its gradients are ready. This model does not simulate those runtime mechanisms.

## Example: reference dump

These values are printed by the example test. The five gradient sizes are respectively 3, 2, 4, 1, and 2 MiB, expressed below in bytes.

```text
param_sizes = [3145728, 2097152, 4194304, 1048576, 2097152]
bucket_cap_mb = 5.0, world_size = 2
ready_times = [[1.0, 2.0, 4.0, 3.0, 8.0], [2.0, 3.0, 6.0, 1.0, 5.0]]

{
  'buckets': [[4, 3], [2], [1, 0]],
  'bucket_ready_times': [[8.0, 4.0, 2.0], [5.0, 6.0, 3.0]],
  'global_ready_times': [8.0, 6.0, 3.0],
  'enqueue_times': [8.0, 8.0, 8.0]
}
```

Bucket 1 must wait until rank 1's time 6 to be globally ready, but still cannot pass bucket 0, whose rank 0 is ready only at time 8. Bucket 2 is ready even earlier and is blocked by the same order constraint.
