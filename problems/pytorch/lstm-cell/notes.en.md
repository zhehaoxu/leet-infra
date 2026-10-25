# LSTM Cell in Detail

## What the three gates do

Think of the cell state $c$ as a "conveyor belt", and the gates as **element-wise differentiable switches** acting on it ($\sigma$ outputs values in $[0,1]$):

- **Forget gate** $f$: decides how much of the previous memory $c_{t-1}$ is retained → $f\odot c_{t-1}$.
- **Input gate** $i$ and **candidate** $g=\tanh$: decide how much is newly written → $i\odot g$.
- **Output gate** $o$: decides how much is read out from the updated memory → $h_t=o\odot\tanh(c_t)$.

The addition in $c_t=f\odot c_{t-1}+i\odot g$ (rather than the plain RNN's $\tanh(W h + U x)$) is the key: gradients can flow back almost losslessly through the "$\oplus$" to much earlier steps, mitigating vanishing gradients.

## Gate layout convention `[i, f, g, o]`

PyTorch `LSTMCell` concatenates the four gates along the channel dimension of `gate`, in the fixed order **i, f, g, o** (CUDNN uses the same layout). After `chunk(4, dim=1)` the chunks must be unpacked under the matching names; writing `i, g, f, o` instead swaps "candidate/forget" and makes every result wrong without raising an error — this is the most classic silent bug when implementing RNNs from scratch.

## Bias initialization trick

Empirically, initializing the **forget gate bias to a positive value** (e.g. +1 to +2) so that $f$ starts close to 1 and leans toward "remembering" can significantly improve training on long sequences (Saxe & Harris 2013). The other gates keep a zero initialization.

## From LSTMCell to sequences

`nn.LSTM` = looping this Cell over time steps + stacking layers + bidirectional concatenation + packed variable-length sequences. Understanding the single step is enough to read `for t in range(T): h, c = cell(x[t], (h, c))`, and it is also the template for implementing custom recurrent units (such as minLSTM, URNN, or cells with residual connections).

## Modern standing

Although Transformers have replaced LSTMs in most sequence tasks, LSTMs are still active in: sequence modeling with small data / low compute, temporal control (such as RL recurrent policies), and as a "gating inductive bias" that inspires new architectures (xLSTM and the gated decay of RetNet both carry LSTM's shadow).
