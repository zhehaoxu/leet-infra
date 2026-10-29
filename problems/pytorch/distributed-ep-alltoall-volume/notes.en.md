## Deriving network load from expert routes

For every expert \(e\) at source rank \(r\), the destination is fixed at \(q=e\bmod W\). Accumulating the token count into \(S_{rq}\) gives the send matrix. The receiver's perspective swaps source and destination, so \(R_{qr}=S_{rq}\). The expert count need not be divisible by \(W\), and some ranks may own no experts.

The count matrices retain local routes to describe all token destinations, but network bytes come only from \(q\ne r\). Rank \(r\)'s network send tokens are its row sum minus \(S_{rr}\); its network receive tokens are its column sum minus \(S_{rr}\). Adding and multiplying by bytes per token gives

\[
B_r=b\left(\sum_q S_{rq}+\sum_q S_{qr}-2S_{rr}\right).
\]

Consequently, \(\sum_r B_r=2b\sum_{r\ne q}S_{rq}\). The factor of 2 counts the sender and receiver endpoints, not two operations for dispatch and combine. Even a very large local count does not change any rank's network load.

## Imbalance, bandwidth, and precision

Let \(M\) be the maximum load and \(\mu\) the mean across all ranks, including idle ones. Imbalance \(M/\mu-1\) is zero for uniform load. Define it as zero for all-zero load to avoid division by zero. For example, if only `0 -> 1` carries traffic among four ranks, loads are `[x, x, 0, 0]` and imbalance is 1; idle ranks must not be removed from the mean.

Counts and byte accounting use int64. The reference computes the mean in float64 rather than first casting to float32. Two identical loads of \(2^{24}+1\), for example, would otherwise round and incorrectly yield nonzero imbalance. The final two metrics are floating-point values and remain subject to float64 representation limits.

Bandwidth means each rank's aggregate network send+receive bytes/second. The single-dispatch time estimate is maximum load divided by this bandwidth. Multiplying bytes per token by \(k\) multiplies bytes and time by \(k\) without changing imbalance; multiplying bandwidth by \(k\) divides time by \(k\). Do not substitute a full-duplex `max(send, receive)` calculation or add another combine operation.

## Constraints and boundaries

The starter specifies nonnegative int64 routing, \(W\ge1\), \(E\ge0\), positive bytes per token, and finite positive bandwidth. Counts and intermediate integer operations must fit int64. The implementation is evaluated on valid inputs; additional invalid-input validation is not required. With \(E=0\), no expert routes exist, but full zero count matrices are still returned.

When \(W=1\), every route is local. All-local or all-zero routing gives zero network bytes, maximum load, imbalance, and time. Asymmetric routing does not imply equal sends and receives, although the receive matrix is always the send matrix's transpose. Output tensors retain the input device.

## Complexity and independent verification

The reference scans an expert mask for every source/destination rank pair. Time is \(O(W^2(E+1))\), including matrix initialization and reductions; space is \(O(W^2+E)\), excluding the input. Direct expert accumulation or scatter could reduce time to \(O(WE+W^2)\), but this is not part of the required semantics.

The independent tests iterate over each source and expert, adding every nonlocal route's bytes directly to both endpoints instead of copying the reference's row/column-sum formula. Coverage includes odd expert counts, fewer experts than ranks, noncontiguous tensors, all-local routes, asymmetry, an off-diagonal straggler, large-integer means, and byte/bandwidth scaling.

## Simplified model versus real systems

This is a payload ledger for one dispatch, not an actual collective execution. It excludes expert computation and latency, communication startup latency, metadata, topology, and link contention. It does not simulate token dropping or expert capacity limits, and it excludes reverse combine. The `e % W` placement is a problem convention, not a claim about every EP system. The time is an estimate under the aggregate-bandwidth assumption, not a measurement of a real training step.
