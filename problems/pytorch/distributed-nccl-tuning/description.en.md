# NCCL Algorithms and Protocols: Cost Selection and Crossover Thresholds

## Calibrated cost selection

Communication choices trade startup latency against sustained bandwidth. A low-latency choice can win for small messages while a higher-bandwidth choice wins for larger ones. Select among these candidates in fixed index order:

**0: Ring-LL · 1: Ring-LL128 · 2: Ring-Simple · 3: Tree-LL · 4: Tree-LL128 · 5: Tree-Simple.**

Use the supplied calibration table, not hardware-independent constants. Each candidate's effective latency \(\alpha_i\) and algorithm bandwidth \(B_i\) already incorporate topology and world-size effects. Do **not** add ring-step counts, tree depths, or other algorithm multipliers.

For message size \(S\) bytes, the model time in microseconds is

\[
 t_i(S)=\alpha_i+\frac{S}{1000B_i}.
\]

Bandwidth is in **decimal GB/s**, not GiB/s or gigabits/s. Record every candidate's time and choose the minimum; exact ties go to the smallest candidate index. Preserve the original message order.

## Crossings and throughput

Let \(\beta_i=1/(1000B_i)\). For each pair with unequal slopes, solve the equal-cost equation:

\[
 S_{ij}=\frac{\alpha_j-\alpha_i}{\beta_i-\beta_j}.
\]

Keep nonnegative real crossings without rounding to whole bytes. Negative crossings, parallel or identical lines, and diagonal entries are `None`. The crossing table is symmetric. An empty message sequence still has a crossing table.

A pairwise crossing is **not necessarily a global optimal switch**: a third candidate may be faster than both lines. Only the lower envelope of all six cost lines determines the winner.

For the winning candidate \(w\), compute effective throughput and utilization:

\[
 A(S)=\frac{S}{1000t_w(S)},\qquad U(S)=\frac{A(S)}{B_w}.
\]

These measure modeled payload throughput and its ratio to the selected algorithm bandwidth, not physical link utilization. At zero bytes, compare startup latencies; throughput and utilization are both zero.

## Example

The numbers below are illustrative calibration values, not universal hardware measurements.

```python
message_sizes = [0, 8000, 16000]
latencies_us = [[1.0, 4.0, 7.0], [4.0, 7.0, 10.0]]
bandwidths_gbps = [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]]
```

Actual reference-implementation output, formatted as JSON (`null` means Python `None`):

```json
{
  "times_us": [
    [1.0, 4.0, 7.0, 4.0, 7.0, 10.0],
    [9.0, 8.0, 9.0, 12.0, 11.0, 12.0],
    [17.0, 12.0, 11.0, 20.0, 15.0, 14.0]
  ],
  "best": [0, 1, 2],
  "thresholds_bytes": [
    [null, 6000.0, 8000.0, null, 12000.0, 12000.0],
    [6000.0, null, 12000.0, 0.0, null, 24000.0],
    [8000.0, 12000.0, null, 4000.0, 0.0, null],
    [null, 0.0, 4000.0, null, 6000.0, 8000.0],
    [12000.0, null, 0.0, 6000.0, null, 12000.0],
    [12000.0, 24000.0, null, 8000.0, 12000.0, null]
  ],
  "achieved_gbps": [0.0, 1.0, 1.4545454545454546],
  "utilization": [0.0, 0.5, 0.36363636363636365]
}
```

At 8000 bytes, Ring-LL and Ring-Simple cross at 9 microseconds, but Ring-LL128 takes only 8 microseconds. Their crossing does not switch the global winner. The actual lower-envelope boundaries here are 6000 and 12000 bytes, with the smaller index winning at each boundary.

## Scope and upstream connection

This is an **NCCL-inspired educational calibrated alpha-beta model**, not a reproduction or reliable predictor of actual NCCL selection. The verified [`ncclTopoGetAlgoTime` in NCCL v2.23.4-1](https://github.com/NVIDIA/nccl/blob/v2.23.4-1/src/graph/tuning.cc) uses `lat * latCount + nBytes / (1000 * bw)`, with latency in microseconds and bandwidth in GB/s. Upstream also applies collective-, size-, topology-, and pipeline-dependent adjustments. Here those details are deliberately replaced by the supplied effective calibration, held fixed across the messages. No GPU, NCCL installation, or additional dependency is needed; the starter contains the data contract.
