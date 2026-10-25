# GRU Explained

## Two gates, one candidate

GRU trims LSTM's input/forget/output gates down to **two gates**:

- **Reset gate** $r=\sigma(W_r x + U_r h)$: decides how much old memory to keep "when computing the candidate new state".
  $r\to0$ means ignoring the history and restarting directly from the current input.
- **Update gate** $z=\sigma(W_z x + U_z h)$: performs a convex interpolation between the old state $h$ and the candidate $\tilde n$,
  playing the combined role of LSTM's "forget + input" gates.
- **Candidate** $\tilde n=\tanh(W_n x + r\odot U_n h)$: note that $r$ acts only on the **state term** $U_n h$
  (PyTorch implements this as `r * nh` after `gi.split`, rather than multiplying $r$ into the entire $gh$ first).

$$ h' = (1-z)\odot \tilde n + z\odot h $$

Large $z$ → keep more of the old $h$; small $z$ → write more of the candidate. That is exactly why it is called the "update gate".

## Aligning with the PyTorch weight layout

`nn.GRUCell.weight_ih` is $(3H, I)$, stacked vertically in three segments $[r; z; n]$; `weight_hh` is likewise $(3H,H)$.
So the right approach is to compute `gi = x@w_ih.T + b_ih` once (giving $(B,3H)$) and then `.split(H, 1)` it into three segments,
rather than slicing the weights up front — this way two matrix multiplies are enough.

## Common pitfalls

- **Gate order**: PyTorch uses **(reset, update, new)**. Writing it as (update, reset, new) gives wrong results;
  this dump was specifically cross-checked against `nn.GRUCell` to pin the order.
- **Where $r$ is multiplied**: `tanh(gi_n + r * gh_n)` — $r$ multiplies only the new segment of $gh$, not the new segment of $gi$.
- **Number of weight elements**: `w_ih` is $3H\times I$ and `w_hh` is $3H\times H$ (the dump once had the wrong arange size, causing view to fail).

## Applications

GRU has 25% fewer parameters than LSTM (one gate fewer) and often matches or beats it on small datasets / short sequences; it is widely used in
speech, time-series forecasting, and machine-translation encoders. Stacking multiple GRUCells over time steps gives `nn.GRU`; bidirectional/multi-layer variants wrap it.
