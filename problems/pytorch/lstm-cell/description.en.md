# LSTM Cell (Single-Step Forward)

## Problem

LSTM solves the vanishing gradient problem of a plain RNN with a "cell state $c$ + three gates": the forget gate decides how much old memory to keep, the input gate decides how much new information to write, and the output gate decides how much to read out. Implementing the forward pass of **one time step** is the foundation for understanding "gating = a differentiable element-wise switch" as well as GRU / gated recurrence / even attention gating.

## Rules

- Implement `solve(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh)`, aligned with `torch.nn.LSTMCell`:
  - `x` has shape `(B, input_size)`, `h_prev/c_prev` have shape `(B, hidden)`;
  - `w_ih` has shape `(4*hidden, input)`, `w_hh` has shape `(4*hidden, hidden)`, `b_ih/b_hh` have shape `(4*hidden,)`.
- Gating: $\text{gate} = xW_{ih}^\top + hW_{hh}^\top + b_{ih} + b_{hh}$, split into 4 chunks in **[i, f, g, o]** order.
  $$ i=\sigma,\ f=\sigma,\ g=\tanh,\ o=\sigma $$
- State update: $c' = f\odot c + i\odot g$, $h' = o\odot\tanh(c')$.
- Return `(h_new, c_new)`, both of shape `(B, hidden)`.

## Examples

```
Input: let w_ih=w_hh=0 and b_ih=b_hh=0 (so all four gate pre-activations are 0), c_prev = 2
  all gates 0 → i=f=o=σ(0)=0.5, g=tanh(0)=0
  c' = 0.5*2 + 0.5*0 = 1.0
  h' = 0.5*tanh(1.0) ≈ 0.3808
Output: h_new ≈ 0.3808, c_new = 1.0
```

## Constraints

- float32 input; `hidden = w_hh.shape[1]`
- Must match `torch.nn.LSTMCell` (with the same weights) within `atol=1e-6`
