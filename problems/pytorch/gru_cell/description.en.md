# GRU Cell (GRUCell)

## Problem

The Gated Recurrent Unit (GRU) uses two gates — the **reset gate** $r$ and the **update gate** $z$ — to control how the
history state is written into the new state. It has one gate fewer than LSTM yet often performs comparably, and is a basic building block
for sequence modeling. This problem implements the GRU recurrence for a **single time step**.

## Rules

- Implement `solve(x, h, w_ih, w_hh, b_ih, b_hh)`; input `x` has shape `(B, input_size)` and hidden state `h` has shape `(B, hidden_size)`.
- Weights follow the PyTorch `nn.GRUCell` convention: `w_ih` has shape `(3*hidden_size, input_size)`,
  `w_hh` has shape `(3*hidden_size, hidden_size)`, and `b_ih` / `b_hh` have shape `(3*hidden_size,)`;
  the three row blocks are ordered **reset, update, new** (denoted $r, z, n$).
- Computation ($\sigma$ is the sigmoid):
  $$ gi = x\,w_{ih}^{\top} + b_{ih},\quad gh = h\,w_{hh}^{\top} + b_{hh} $$
  $$ r = \sigma(gi_r + gh_r),\quad z = \sigma(gi_z + gh_z),\quad \tilde n = \tanh(gi_n + r \odot gh_n) $$
  $$ h' = (1 - z)\odot \tilde n + z \odot h $$
- The result must match a single-step forward of `nn.GRUCell(...)` (with its weights set to this input).

## Examples

```
Input: input_size=2, hidden_size=3
     x = [[0.5, -0.5]],  h = [[0.2, -0.3, 0.1]]
     w_ih = arange(18).reshape(9,2)/20 - 0.4
     w_hh = arange(27).reshape(9,3)/20 - 0.4
     b_ih = b_hh = 0
Output: h' = [[0.084566, -0.161684, 0.035316]]
```

## Constraints

- float32 input; the first dimension of `w_ih`/`w_hh` must be `3*hidden_size`
- The gate block order is fixed as (reset, update, new), consistent with `nn.GRUCell`
- Do not use `nn.GRUCell` / `nn.GRU` / the recurrent operators in `F`; write the gating formula by hand
