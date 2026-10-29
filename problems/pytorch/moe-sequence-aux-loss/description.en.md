# Sequence-Level Load-Balancing Loss

## Task

Implement `solve`: given per-token expert affinities and **supplied** top-k selections, compute expert load and mean probabilities within each sequence, then the sequence-level auxiliary loss. Also return the loss computed after pooling the batch to expose the difference in averaging order.

Compute loss values only: no expert selection, gradients, or optimizer updates. This is not a CV loss on total expert importance or Switch statistics treating the batch as one long sequence.

## Inputs

- `scores`: CPU float64 tensor `[B,T,E]` containing **nonnegative affinities, not logits**. Values lie in `[0,1]` and each expert row sums to at least `1e-6`. Softmax probabilities or sigmoid affinities can be supplied.
- `selected_ids`: CPU int64 tensor `[B,T,K]`. Each token has K distinct IDs in `[0,E)`.
- `alpha`: Python float in `[0,2]`, default `1.0`.
- `B=1..4`, `T,E=1..8`, `1 <= K <= E`. All sequences have the same valid length T, with no padding.
- All inputs are valid. Accept noncontiguous tensors and mutate no inputs.

Selections may come from biased or otherwise constrained routing. They are **not necessarily the top-k scores**; do not recompute them.

## Exact formulas

First normalize each token across all E experts, not only the selected experts:

$$
p_{b,t,e}=\frac{s_{b,t,e}}{\sum_{j=0}^{E-1}s_{b,t,j}}.
$$

Let $c_{b,e}$ count selections of expert e in sequence b. Compute:

$$
f_{b,e}=\frac{E}{TK}c_{b,e},\qquad
P_{b,e}=\frac{1}{T}\sum_{t=0}^{T-1}p_{b,t,e}.
$$

Thus $\sum_e f_{b,e}=E$ and $\sum_e P_{b,e}=1$. The f values are relative balanced loads, not a distribution summing to 1.

$$
\ell_b=\alpha\sum_e f_{b,e}P_{b,e},\qquad
L_{\mathrm{seq}}=\frac1B\sum_b\ell_b.
$$

The comparison term averages across sequences before multiplying:

$$
L_{\mathrm{batch}}=\alpha\sum_e
\left(\frac1B\sum_b f_{b,e}\right)
\left(\frac1B\sum_b P_{b,e}\right).
$$

These are generally different. Do not multiply f or P by `alpha`; statistics must still be returned correctly when `alpha=0`.

## Return value

Return the five-item `tuple` `(f, p, per_sequence, sequence_loss, batch_loss)`.

- `f`, `p`: shape `[B,E]`, representing $f,P$.
- `per_sequence`: shape `[B]`, representing $\ell$.
- `sequence_loss`, `batch_loss`: zero-dimensional tensors representing $L_{\mathrm{seq}},L_{\mathrm{batch}}$.
- All five items are CPU float64 tensors, compared with `rtol=2e-11, atol=2e-11`.

## Example

```python
scores = [[[0.9, 0.1], [0.9, 0.1]],
          [[0.1, 0.9], [0.1, 0.9]]]
selected_ids = [[[0], [0]], [[1], [1]]]
alpha = 0.1
```

Output shown as tensor contents:

```python
f = [[2.0, 0.0], [0.0, 2.0]]
p = [[0.9, 0.1], [0.1, 0.9]]
per_sequence = [0.18, 0.18]
sequence_loss = 0.18
batch_loss = 0.1
```

The batch appears balanced, but each sequence concentrates on one expert. If `K=E`, all f entries are 1 and both losses equal `alpha`.
