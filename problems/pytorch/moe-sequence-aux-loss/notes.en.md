# Reference Approach: Multiply Within a Sequence, Average Across Sequences

## Implementation

1. Normalize `scores` across experts to obtain per-token expert probabilities; do not normalize only selected entries.
2. Count selections into `[B,E]` with `scatter_add_` and multiply by `E/(T*K)` for relative load f.
3. Average over the token dimension to obtain p.
4. For each sequence, compute `alpha * sum(f*p)`, then average across the batch for `sequence_loss`.
5. Also compute `batch_loss`: average f and p across the batch separately, then take their dot product.

Selections come from external input and must not be recomputed as top-k of scores, because biased or constrained routers may produce different choices.

## Correctness checks

The 15 `check_*` functions cover the two averaging orders, the single-sequence reduction, per-token normalization, row-scale invariance, K normalization, all experts, `alpha=0`, one expert, explicit selections, unselected experts, sequence permutation, noncontiguous inputs, 32 randomized enumeration cases, the maximum shape, default alpha, and very small affinities.

The hand example deliberately makes the batch look balanced while each sequence concentrates on one expert: `sequence_loss=0.18` and `batch_loss=0.1`, catching a reversed averaging order. The oracle counts selections and probabilities element by element in pure Python and does not reuse the scatter path.

## Scope and sources

Public DeepSeek-V2/V3 descriptions include a sequence-wise auxiliary loss whose relative load is often written as $f_i=\frac{E}{Tk}\sum_{t}\mathbb 1[i\in \mathrm{topk}(u_t)]$. This problem follows that functional form and places "multiply within a sequence, then average across the batch" next to "average the batch first, then multiply" to expose implementation order.

The problem uses supplied selections and bounded affinities; it implements no router, training, CV loss, or expert-importance statistic. It is not a step-by-step reproduction of any framework training loop. The todo previously associated sequence-level loss with the OLNN/CV paper, but CV is a different statistic, so this problem does not adopt that paper as its specification.
