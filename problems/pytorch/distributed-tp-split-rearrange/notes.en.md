## 1. Split the computation, not just the storage

Weights use the PyTorch layout `[out_features, in_features]`. Column parallel therefore slices weight **rows** (axis 0), despite its name: the output feature range belongs to a rank. Each local matrix product produces `[B, O/P]`, and concatenating along the feature axis restores `[B, O]` in global feature order.

Row parallel slices weight **columns** and matching input features. Partitioning the contraction index gives

\[
xW^{\mathsf T}=\sum_{r=0}^{P-1}x^{(r)}(W^{(r)})^{\mathsf T}.
\]

Every partial product already has shape `[B, O]`. Concatenating them, averaging them, or taking only one rank is incorrect. The reference accumulates them with SUM. Splitting/reconstructing the original weight and then calling one dense matrix multiplication would miss the intended computation.

## 2. Vocabulary ownership and stable CE

With shard width `V // P`, integer division identifies the target's owner and remainder identifies its local position. Return mappings for **every** vocabulary ID, not just the labels present in the batch. After a row permutation, IDs refer to the new row order; consistently permuting rows and relabeling targets must preserve losses.

First compute each rank's local logits. A per-sample MAX across local maxima supplies a common shift. Subtract it before exponentiation, sum exponentials locally, then SUM the resulting batch vectors across ranks. Independent local softmax denominators are wrong: normalization must cover the whole vocabulary.

For the numerator, each rank gathers at a safe local index and masks nonowned samples to zero before SUM. A clamped index is only a way to avoid an out-of-range read on a nonowner; it is not a replacement label. Masking must happen before reducing contributions.

Use `log(global_exp_sum) - shifted_target_logit`. This is algebraically the usual log-sum-exp minus the target, but avoids adding a huge shared offset back and losing the small logarithmic term through cancellation. A sample with all logits equal to `1e16` still has loss `log(V)`. Positive and negative logits around `10000` also require shifting to avoid overflow or underflow. Do not modify logits or inputs through shard views in place.

## 3. Cost and communication interpretation

Let `B` be batch size, `I` input width, `O` output width, and `P` rank count (`O=V` in vocab mode).

- Matrix products total `O(B*O*I)` arithmetic across simulated ranks; each evenly sized rank has about one `P`-th of this work.
- Column gathers `B*O` output values. Row sums `P` partial `[B, O]` results, adding `O(P*B*O)` reduction work.
- Vocab adds `O(B*V)` elementwise work plus rank reductions of batch vectors. Conceptually it needs a MAX and two SUM reductions, not a vocabulary-sized gather for CE. The gathered logits are required here only as an inspectable output.
- Excluding inputs and view headers, column/row need `O(B*O)` tensor workspace. This single-process vocab reference retains local and gathered logits and stacked local maxima, needing `O(B*V + B*P)` tensor space plus `O(V)` for the ownership map. Weight shards are views; copying them is permitted but adds weight storage.

These are simulation work/storage bounds, not network latency or collective bandwidth estimates.

## 4. Scope and checks

The exercise covers forward split/gather/reduce equivalence and unreduced vocab CE on CPU, not process groups, NCCL, bias handling, embedding lookup, padding an uneven vocabulary, backward communication, or production memory optimization. Invalid modes and nondivisible split dimensions explicitly raise `ValueError`; other inputs are guaranteed by the contract.

Tests compare directly with `x @ W.T` and dense PyTorch CE as independent numerical references. They exercise non-square matrices, one rank, one class, shard-edge labels, relabeling, float64 extreme offsets, float32/noncontiguous inputs, exact shard contents/dtypes, and input preservation. Dense CE is appropriate in the test oracle, not in the submitted vocab implementation.
