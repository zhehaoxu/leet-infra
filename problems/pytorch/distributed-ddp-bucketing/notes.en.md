## 1. Separate membership from availability

Construct bucket membership once, using reverse parameter indices and the stated byte-cap rule. This order is a toy convention, not a claim that autograd always completes gradients in reverse registration order. Branches, shared parameters, and the execution graph can produce different readiness patterns; the input times are authoritative here.

Maintain a current bucket and its total bytes. If the next gradient would overflow a nonempty bucket, seal it first. Equality fits. An individually oversized gradient is immediately sealed as a singleton, so neither a preceding partial bucket nor a following gradient can share it. Do not create empty buckets or sort members by size/time.

`bucket_cap_mb` uses MiB: multiply by `1024**2`, then truncate the positive value to an integer. Rounding to nearest or using decimal megabytes changes bucket boundaries. Parameters remain indivisible. The contract guarantees positive integer sizes, a positive converted cap, a positive rank count, and finite nonnegative readiness times; no additional invalid-input behavior is promised.

## 2. Three different time summaries

For a rank-local bucket, take the maximum readiness of its member parameters. Taking the first or last visited member's time is incorrect, even though membership has a fixed order.

For global readiness, take the maximum over **all** ranks for that bucket. The latest rank can differ between buckets; there is no representative rank or rank assigned to a bucket. With `K` buckets and `W` ranks, the local readiness matrix is `[W, K]`, not `[K, W]`.

Finally scan global readiness left to right, retaining the largest value seen. This prefix maximum is the earliest globally synchronized eligibility compatible with the common bucket order. For example, global times `[1, 9, 5, 3]` become `[1, 9, 9, 9]`: buckets 2 and 3 cannot bypass bucket 1. It is neither a prefix sum nor a sort. Equal enqueue times do not imply zero-duration network completion; duration simply is not modeled.

For empty parameters, there are no maxima to take. Return no buckets, no global/enqueue times, and one empty local-readiness row per rank.

## 3. Why there is no start-owner rank

All-reduce combines contributions from every rank and makes the reduced result available to every rank. It has no designated root like a rooted reduce or broadcast operation. Ring all-reduce distributes chunk work across ranks; it does not make bucket `k` belong to rank `k % world_size`.

Thus five buckets on two ranks still involve both ranks in all five reductions. Changing rank count or readiness can change eligibility times, but not this exercise's size-only bucket membership.

## 4. Honest runtime scope

The model's `enqueue_times` are global synchronized eligibility times, not exact per-process calls into a real reducer. A process may enqueue a collective asynchronously before a peer is ready. Matching collective order still matters, but a real implementation need not impose the global barrier suggested by this educational summary.

Real DDP may rebuild buckets from observed behavior, group tensors by dtype/device, and overlap asynchronous synchronization with backward computation. Bucket capacity describes membership, not a trigger meaning "full bytes implies ready gradients." This exercise has no dtype/device partitioning, communication duration, network topology, bandwidth model, or communication/computation overlap simulation.

## 5. Complexity and independent checks

With `N` parameters and `K <= N` buckets, packing costs `O(N)`. Rank-local readiness examines every parameter on every rank in `O(W*N)`; global maxima cost `O(W*K)` and the prefix scan costs `O(K)`. Total time is `O(W*N + W)`, including constructing `W` empty rows when `N=0`. Returned storage is `O(N + W*K + W)`; the reference uses constant additional scalar bookkeeping beyond the result lists.

The test oracle deliberately does not copy the streaming packer: it enumerates possible contiguous prefix endpoints using cumulative byte sums and chooses the longest legal prefix. Expected enqueue times are computed directly from all underlying gradients in all preceding buckets, rather than copying the running-maximum recurrence. Tests include deliberately non-reversed readiness, late ranks, blocked later buckets, exact capacity, middle oversized items, fractional MiB/flooring, zeros, empty inputs, one rank, and five buckets on two ranks.
