# xLSTM sLSTM: the max stabilizer, normalizer state, and scalar-memory pointwise recurrence ledger

## Background

The **sLSTM** (scalar LSTM) of the xLSTM paper sits next to mLSTM: it uses no matrix memory, but gives every channel a scalar memory `c`, pairs it with a **normalizer state** `n` that acts as the denominator, and uses a **max stabilizer** `m` to keep the exponential gates inside a numerically safe region.

The official `xlstm/blocks/slstm/src/vanilla/slstm.py` writes this step compactly:

- `m` is a per-channel scalar stabilizer that guarantees both gates stay at most 1;
- `n` is the normalizer state, accumulating the write amounts and finally serving as the denominator of the output;
- the first frame (all of `n` zero) takes a special branch.

This task rebuilds these three **pure numeric** ledgers: the `max` stabilizer with its two gates, the pointwise recurrence of the scalar memory plus normalizer state, and the two initializations of the `f`-gate bias. GEMM layouts, the CUDA kernel and autograd are all out of scope — every input is a list of already-computed numbers.

## General conventions

- All inputs are lists of Python `float`; no tensors, no gradients, no batching.
- Floating point comparisons use absolute tolerance `1e-9` plus relative tolerance `1e-9`.
- `head_dim` is written as `H`, i.e. the number of channels.
- Every mode returns a `dict`; the field names are listed per mode below.
- Indices are zero-based and **inclusive** on both ends.

## Modes and rules

### 1. `max_stabilizer`: the max stabilizer and the two gates

Inputs:

| Parameter | Meaning |
|---|---|
| `iraw` | input gate pre-activation, length `H` |
| `fraw` | forget gate pre-activation, length `H` |
| `m_prev` | max stabilizer of the previous frame, length `H` |
| `n_prev` | normalizer state of the previous frame, length `H` |

Steps:

```
logfplusm[i] = m_prev[i] + logsigmoid(fraw[i])
zero_n_branch = all n_prev[i] == 0          # frame-level decision, not per element
if zero_n_branch:
    m_new[i] = iraw[i]
else:
    m_new[i] = max(iraw[i], logfplusm[i])
igate[i] = min(exp(iraw[i] - m_new[i]), 1)
fgate[i] = min(exp(logfplusm[i] - m_new[i]), 1)
```

Returned fields:

- `logfplusm`: length `H`;
- `m_new`: length `H`;
- `zero_n_branch`: boolean, whether the all-zero branch was taken;
- `igate` / `fgate`: length `H`;
- `gate_le_one`: a boolean list of length `H`, per channel indicating whether both gates stay at most 1.

Pitfalls:

- `logfplusm` uses `logsigmoid(fraw)`, not `sigmoid(fraw)`;
- `m_prev` must take part in the sum forming `logfplusm`;
- **`zero_n_branch` is a frame-level decision**: as soon as one channel has a non-zero `n_prev`, the whole frame takes the `max` branch and every channel is handled uniformly;
- both gates are hard-clipped by `min(·, 1)`;
- `logsigmoid(x)` is defined as `-softplus(-x)`, that is `-log(1 + exp(-x))` when `x >= 0` and `x - log(1 + exp(x))` otherwise.

### 2. `pointwise_step`: scalar memory plus normalizer state recurrence

Inputs:

| Parameter | Meaning |
|---|---|
| `wx` | input-side projection, length `4H` |
| `ry` | recurrent-side projection, length `4H` |
| `bias` | gate bias, length `4H` |
| `states` | four lists of length `H`, in the order `y` / `c` / `n` / `m` |
| `head_dim` | the number of channels `H` |

Steps:

```
raw = wx + ry + bias                                  # length 4H
iraw, fraw, zraw, oraw = split into [i, f, z, o] groups of H each
logfplusm / m_new / igate / fgate      # same as max_stabilizer
ogate[i] = sigmoid(oraw[i])
zval[i]  = tanh(zraw[i])
c_new[i] = fgate[i] * c_prev[i] + igate[i] * zval[i]
n_new[i] = fgate[i] * n_prev[i] + igate[i]
y_new[i] = ogate[i] * c_new[i] / n_new[i]
```

Returned fields:

- `raw_gates`: four lists of length `H`, in the order `iraw` / `fraw` / `zraw` / `oraw`;
- `logfplusm` / `m_new` / `igate` / `fgate`: length `H`;
- `zero_n_branch`: boolean;
- `ogate` / `zval` / `c_new` / `n_new` / `y_new`: length `H`.

Pitfalls:

- the gate order in `raw` is `[i, f, z, o]`, each group occupying `H` consecutive entries;
- `tanh` applies **only to `zraw`** (`igate * tanh(zraw)`), not to `c_new`;
- the denominator of `y_new` is `n_new` (the normalizer state), not `c_new`;
- the output is **not** wrapped in `tanh`;
- when `n_new[i] == 0`, `y_new[i]` is recorded as `NaN` (the IEEE 754 `0/0` semantics).

### 3. `bias_powerlaw`: the `f`-gate bias initialization

Inputs:

| Parameter | Meaning |
|---|---|
| `head_dim` | the number of channels `H` |
| `block_idx` | index of the current block, starting from 0 |
| `num_blocks` | total number of blocks |
| `bias_mode` | optional, `powerlaw` (default) or `small_init` |

`powerlaw` mode:

```
ratio = block_idx / (num_blocks - 1)   if num_blocks > 1 else 0.0
exponent = 0.3 + 1.3 * ratio
bias[d] = -( -5.0 + 12.0 * (d / (H - 1)) ** exponent )      # d = 0 .. H-1
```

`small_init` mode:

```
bias[d] = linspace(3.0, 6.0, H)[d]      # endpoints included
```

Returned fields: `bias` (length `H`), `exponent` (float), `ratio` (float).

Pitfalls:

- **the minus sign sits outside the parentheses**: `bias = -( -5 + 12 * base^exponent )`, so with `base = 0` you get `bias = 5`;
- `exponent` grows with the block index (from `0.3` to `1.6`);
- when `num_blocks == 1`, `ratio = 0.0`;
- when `H == 1`, `d / (H - 1)` is `0 / 0`, so in `powerlaw` mode `bias` is `NaN`;
- in `small_init` mode with `H == 1` you get `[3.0]`, and `exponent` is recorded as `0.0`.

## Input and output examples

`max_stabilizer`:

```python
solve("max_stabilizer", iraw=[1.0, -2.0], fraw=[0.5, 0.25],
      m_prev=[0.0, 0.0], n_prev=[0.0, 0.0])
# {
#   "logfplusm":     [-0.4741, -0.5759],
#   "m_new":         [1.0, -2.0],          # all-zero branch -> m_new = iraw
#   "zero_n_branch": True,
#   "igate":         [1.0, 1.0],
#   "fgate":         [0.2290, 1.0],        # channel 1 raw value 4.154 clipped to 1
#   "gate_le_one":   [True, True],
# }
```

`pointwise_step`:

```python
solve("pointwise_step", wx=[0.0, 0.0, 2.0, 0.0], ry=[0.0, 0.0, 0.0, 0.0],
      bias=[0.0, 0.0, 0.0, 0.0],
      states=[[0.0], [1.0], [2.0], [0.0]], head_dim=1)
# {
#   "raw_gates":        [[0.0], [0.0], [2.0], [0.0]],
#   "logfplusm":        [-0.6931],
#   "m_new":            [0.0],
#   "zero_n_branch":    False,
#   "igate":            [1.0],
#   "fgate":            [0.5],
#   "ogate":            [0.5],
#   "zval":             [0.9640],
#   "c_new":            [1.4640],
#   "n_new":            [2.0],
#   "y_new":            [0.3660],
# }
```

`bias_powerlaw`:

```python
solve("bias_powerlaw", head_dim=4, block_idx=0, num_blocks=1)
# {
#   "bias":     [5.0, -3.6307, -5.6256, -7.0],
#   "exponent": 0.3,
#   "ratio":    0.0,
# }
```

## Out of scope

- the GEMM between the `R` matrix and the state, and the `reshape` / `transpose` layout chain;
- CUDA kernel launch configuration, `round_to_power2`, `SLSTM_FORWARD_CLIPVAL`;
- autograd backward passes and gradient clipping;
- the random initialization of parameters (the `zeros` / `standard` weights, the normal sampling of `small_init_init_`);
- the plain LSTM variant in `vanilla/lstm.py` (`sigmoid` gates, `tanh(c)` output) — kept only as a contrast;
- multi-head splitting and configurations with a `num_gates` other than 4.
