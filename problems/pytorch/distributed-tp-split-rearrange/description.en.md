# Distributed: TP Split/Reduce and Vocab-Parallel Cross Entropy

## Description

Simulate a bias-free tensor-parallel linear layer on one CPU. Splitting and rejoining the **weights alone** is not enough: compute with each shard and reconstruct the forward result. For weights \(W\in\mathbb R^{O\times I}\) and inputs \(x\in\mathbb R^{B\times I}\), every mode must produce the same output as \(xW^{\mathsf T}\).

The starter contains the full input/output contract. Preserve input values and floating dtype; no distributed runtime is needed.

## Split and combine rules

Let \(P\) be the tensor-parallel size. Shards follow contiguous index ranges in rank order.

- **Column:** split the output dimension of \(W\), axis 0. Each rank computes \(x(W^{(r)})^{\mathsf T}\). Concatenate these outputs along axis 1, modeling an output all-gather.
- **Row:** split the input dimension of both \(W\) and \(x\), axis 1. Each rank computes \(x^{(r)}(W^{(r)})^{\mathsf T}\). **Sum**, rather than concatenate, the partial outputs, modeling a SUM all-reduce.
- **Vocab:** treat \(O=V\) as vocabulary size and split \(W\) on axis 0. Compute local logits as in column mode, and additionally compute one cross-entropy loss per sample using the shard reductions below. Concatenated logits are returned for inspection, not used for the loss calculation.

An unsupported mode or a split dimension not divisible by \(P\) must raise `ValueError`. Other inputs satisfy the starter's contract.

## Stable vocab-parallel loss

With \(L=V/P\), global label \(v\) belongs to rank \(\lfloor v/L\rfloor\), at local index \(v\bmod L\). This ownership uses the current vocabulary row order, even if the rows have been permuted.

For sample \(b\), let \(z^{(r)}_{bj}\) be local logits and \(y_b\) the target:

1. Reduce local maxima across ranks: \(m_b=\max_{r,j}z^{(r)}_{bj}\).
2. Reduce local exponent sums: \(s_b=\sum_r\sum_j\exp(z^{(r)}_{bj}-m_b)\).
3. Only the target's owner gathers its shifted target logit; every other rank contributes zero. SUM these masked contributions to obtain \(t_b=z_{b,y_b}-m_b\).
4. Return \(\ell_b=\log s_b-t_b\), without reducing over the batch.

Do not replace these steps with dense `cross_entropy`. The MAX and SUM operations here are CPU simulations of collective reductions.

## Example: reference dump

Inputs use float64; tensors are displayed as lists and losses are rounded to six decimal places. These values are printed by the example test.

```text
W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
x = [[2.0, 1.0], [-1.0, 3.0]]
tp_size = 2, mode = 'vocab', targets = [2, 0]

{
  'shards': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [-1.0, 0.0]]],
  'label_map': {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)},
  'output': [[2.0, 1.0, 3.0, -2.0], [-1.0, 3.0, 2.0, 1.0]],
  'losses': [0.412078, 4.419717]
}
```

The two targets have different owners. Both losses still normalize over **all four** vocabulary entries, not just the target's shard.
