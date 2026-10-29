# Notes: Splitting the LSTM Backward into a Gate Ledger and a Time Ledger

## Origin

"Write a one-layer LSTM forward and BPTT by hand" is a staple whiteboard question for speech/time-series and training-framework roles, and it is a sharp lie detector for whether a candidate actually derived the formulas, because the LSTM backward has **two places that almost everybody gets wrong the first time**: the \(o\) factor in front of \(\tanh(c)\), and the gate slice order.

## The Gate Order Is (i, f, g, o), Not the Textbook (i, g, f, o)

Papers tend to narrate `input, cell-candidate, forget, output`, but the row blocks of PyTorch's `weight_ih_l0` are really `i | f | g | o`. Using `nn.LSTM` as the oracle pins this down: slice the gates wrongly and even the forward \(h\) disagrees with the kernel, so the cross-check fails instantly. The safe route is comparing `c_t` against the kernel output, or trying all four permutations and keeping the one that matches bit for bit.

## The o Factor in dc_t

Since \(h_t=o_t\tanh(c_t)\), \(\partial h_t/\partial c_t=o_t(1-\tanh^2c_t)\) and therefore

$$dc_t = dc_{t+1} + dh_t\odot o_t\odot(1-\tanh^2c_t)$$

Writing only \(dh(1-\tanh^2c)\) is the classic missing factor and it already fails at \(T=1\), so it cannot hide. Likewise \(do=dh\tanh(c)\), not \(dh\cdot c\). When folding into \(dgates\), the sigmoid derivative is \(y(1-y)\) (dropping the leading \(y\) changes the magnitude of every gate gradient) and the tanh derivative is \(1-y^2\).

## The Time Ledger and the Bias Ledger

- \(dx_t=dgates_tW_{ih}\), \(dh_{t-1}=dgates_tW_{hh}\), \(dc_{t-1}=dc_t\odot f_t\): all three come from the same \(dgates_t\), so `dh_next/dc_next` must persist across the loop instead of being reset per step.
- \(dW_{ih}=\sum_t dgates_t^T x_t\) and \(dW_{hh}=\sum_t dgates_t^T h_{t-1}\) **accumulate over time**, with \(h_{-1}=h_0\) included in the second one.
- \(db_{ih}=db_{hh}=\sum_t dgates_t\): both biases are added at full length in `gates`, so the two gradients must be identical (`check_bias_pairing` asserts `got[3] == got[4]`).
- Truncated BPTT: the terminal state gradient may be entirely zero while the per-step contributions still have to propagate back.

## How It Differs from Neighboring Problems

`ag-activation-backward` only deals with elementwise derivatives; the forward-only `lstm-cell` and `gru_cell` never accumulate across time. The signature difficulty here is the pair of facts — gate order and the \(o\) factor — that pure memorization cannot dodge.

## Cross-Check Basis and Complexity

The oracle builds `nn.LSTM(E, Hid).to(xs.dtype)`, `copy_`s the four parameters, passes 3-D `(1, N, Hid)` h_0/c_0 and takes gradients of `loss=(hn*dhT).sum()+(cn*dcT).sum()`, reordering to `(dxs, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0)` with atol=1e-8 in float64. The example is the minimal \(T=N=Hid=1\) case where all seven quantities are hand-verifiable. Complexity is \(O(T\cdot N\cdot E\cdot Hid)\).
