# Reference Approach: Group Selection, Bias Isolation, and Sign Updates

## Implementation

1. `scores = sigmoid(x @ gate_w.T)` and `choice = scores + bias`.
2. Reshape choice into contiguous equal-sized groups and sum each group's two largest values for its group score.
3. Take the `top_groups` groups by descending score; stable sorting gives the lower group index on ties.
4. Mask choice in unselected groups to `-inf`, then take the k largest experts.
5. Build weights from **raw scores only**, normalized over selected entries; bias never enters weighting.
6. Count the entire batch's load with `bincount`; if updating, adjust bias by `sign(average load - actual load)`.

The bias update happens once per batch, not per token; average load is $Tk/E$. With `update_bias=False`, return a `clone` so the caller cannot mutate the input through the result.

## Correctness checks

The 14 `check_*` functions cover a hand example with bias updates, bias affecting selection only, group selection using the top-2 sum, all groups reducing to ordinary top-k, batch-global load updates, returning a copy when disabled, zero gamma, group/expert tie order, k boundaries, saturated inputs, noncontiguous inputs, 36 randomized cases, invalid structure, and the maximum shape.

The oracle computes sigmoid, grouping, masking, selection, normalization, and counts per token in pure Python, independent of torch's `topk`/`argsort`/`scatter`. The hand example decreases bias for overloaded experts and increases it for underloaded experts, catching a reversed sign; the bias example catches mixing bias into weights.

## Scope and sources

The mechanism corresponds to auxiliary-loss-free balancing described in the DeepSeek-V3 technical report: sigmoid gating, per-expert bias used for selection, grouped top-k, and fixed-step bias adjustment based on load. This problem states the routing forward pass and one batch-level update as explicit, reproducible rules.

It does not claim exact agreement with any training framework: real training may update bias at optimizer-step boundaries, use different load-statistic conventions, and reduce across distributed workers; this problem involves no gradients, optimizer, capacity constraint, or all-to-all communication. Tie order, group size, and update sign are fixed by the statement. Other mechanisms mentioned in the todo (dead-expert ledger, soft clamp, expert dropout) are not implemented in this batch and are not mixed into this problem.
