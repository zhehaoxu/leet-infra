# Solution Notes: Calibrated NCCL-Inspired Cost Selection

## 1. Interpret the calibration, then normalize units

Flatten the calibration tables row by row: Ring first, then Tree, with LL, LL128, Simple inside each row. This gives the required candidate indices without algorithm-specific branches. Neither algorithm names nor protocol names imply additional constants: the input already represents effective startup latency and algorithm bandwidth for the intended environment.

Decimal bandwidth \(B\) GB/s is \(B\cdot10^9\) bytes/s. Transferring \(S\) bytes therefore takes

\[
 \frac{S}{B\cdot10^9}\cdot10^6=\frac{S}{1000B}
\]

microseconds. Adding startup latency yields \(t(S)=\alpha+\beta S\), where \(\beta=1/(1000B)\). Do not substitute 1024, multiply by eight, or add Ring/Tree communication factors. In particular, algorithm bandwidth is not automatically bus bandwidth.

## 2. Evaluate the lower envelope directly

For each message, evaluate all six lines and select the smallest time. Scanning candidates in index order and replacing the winner only on a strict improvement gives deterministic tie-breaking. Python's `min` over indices in order has the same behavior. Use exact comparisons for winner selection, not a tolerance: a slightly slower candidate must not win merely because its index is smaller.

The winner is a point on the lower envelope of all cost lines. There is no need to sort messages, assume a fixed LL-to-LL128-to-Simple progression, or use pairwise thresholds to dispatch messages. The supplied table can make any candidate win.

## 3. Derive pairwise crossings separately

Equating two lines gives

\[
 (\beta_i-\beta_j)S=\alpha_j-\alpha_i.
\]

If bandwidths are equal, the slopes are equal. Different intercepts mean no intersection; equal intercepts mean infinitely many intersections. Neither case has a unique threshold, so both use `None`. A candidate's diagonal also uses `None`.

For unequal bandwidths, compute the real crossing and keep it only when nonnegative. The slope difference can be evaluated as

\[
 \beta_i-\beta_j=\frac{B_j-B_i}{1000B_iB_j},
\]

which avoids subtracting two nearly equal reciprocals. The reference uses this rearrangement. Nearly equal bandwidths are not treated as equal with an epsilon; they can produce a large valid threshold. Compute each unordered pair once and assign its value to both matrix entries. Normalize signed zero if necessary, and do not round fractional byte crossings.

A crossing is only an equality between two lines. In the example, candidates 0 and 2 meet at 8000 bytes and 9 microseconds, while candidate 1 takes 8 microseconds. The real optimal regions are candidate 0 through 6000 bytes, candidate 1 above 6000 through 12000 bytes, and candidate 2 above 12000 bytes. The inclusive boundaries follow the index tie rule, not a property of Ring or Tree.

## 4. Derive and interpret the throughput metrics

After selecting \(w\), compute throughput from its full time, including startup:

\[
 A=\frac{S}{1000t_w},\qquad
 U=\frac{A}{B_w}=\frac{S}{1000\alpha_w B_w+S}.
\]

Positive startup latency makes the exact modeled utilization less than one for a finite message and equal to zero for an empty message. As size increases for a fixed candidate, throughput approaches its calibrated bandwidth. Floating-point rounding can make utilization numerically equal to one. Neither metric accounts for other traffic, link encoding, or collective bus-volume amplification, so neither should be reported as measured hardware utilization.

## 5. Edge cases and verification

- At zero bytes, only latency determines the winner. The winning time remains positive, so both throughput divisions are safe and return zero.
- Equal latencies with unequal bandwidths intersect at zero; this is a valid threshold, not `None`.
- Negative intersections lie outside the message-size domain and must be discarded. Parallel and identical lines also have no unique reported crossing.
- Empty messages produce empty per-message results but still require the calibration-dependent threshold matrix. Repeated or unsorted sizes retain their positions.
- The tests use exact `Fraction` arithmetic for integer and binary-fraction fixtures, with an independently cross-multiplied intersection equation. They also substitute reported crossings back into both cost lines, check symmetry, test all six winner indices, distinguish near ties from exact ties, and verify winners immediately before and after genuine lower-envelope switches.
- Scaling byte sizes and bandwidths by the same factor preserves times and winners, scales throughput and byte thresholds by that factor, and preserves utilization. This helps catch decimal-unit and hidden-multiplier mistakes.

## 6. Complexity

With \(N\) messages and \(K\) candidates, time is \(O(NK+K^2)\). The required output occupies \(O(NK+K^2)\) space; auxiliary working space is \(O(K)\) beyond the output. Here \(K=6\), so total time and output space are \(O(N+1)\), and auxiliary space is constant. No hardware access or third-party library is needed.

## 7. What is, and is not, taken from NCCL

The verified upstream reference is [NVIDIA NCCL v2.23.4-1, `src/graph/tuning.cc`](https://github.com/NVIDIA/nccl/blob/v2.23.4-1/src/graph/tuning.cc). In that version, `ncclTopoGetAlgoTime` reads calibrated latency/bandwidth entries and ends with:

```cpp
*time = lat * latCount + nBytes / (1000 * bw);
```

The file explicitly documents latency in microseconds and bandwidth in GB/s. It also handles unavailable bandwidth entries, backup Ring bandwidth, Tree AllReduce size corrections, a multi-node Ring-Simple latency adjustment, and aggregation-dependent latency counts. This exercise borrows only the latency-plus-transfer-time structure: effective latency absorbs the intended startup effects, and the supplied algorithm bandwidth is used directly.

There are no universal hardware constants here, and the illustrative example is not a hardware benchmark. Real NCCL supports additional algorithms, eligibility rules, and version-dependent tuning. A fixed six-candidate educational model does not reproduce NCCL's full selector or establish which implementation will actually run fastest on a machine.
