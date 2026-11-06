# xLSTM mLSTM: stabilized exponential gates, D matrix, and intra-chunk gate ledger

## Background

`xlstm/blocks/mlstm/backends.py` in `NX-AI/xlstm` (`main`) expresses mLSTM in three equivalent forms with pure torch: parallel, recurrent step, and chunkwise. None of them holds real weights — `q / k / v / i / f` are all passed in as arguments, which makes them ideal for **stateless judging**. This problem extracts three distinctive sets of intermediate values into a single entry point `solve(mode, **kw)` with 3 modes; every input is a small numeric list (`DH <= 3`, `S <= 4`, `chunk_size` divides `S`), a pure Python numeric ledger with no `xlstm` import and no real model.

The official `backends.py` on `main` has 265 lines. In `cell.py` the `mLSTMCell` state shapes are `C:(B,NH,DH,DH)` / `n:(B,NH,DH,1)` / `m:(B,NH,1,1)`, and the fgate bias uses `start=3.0, end=6.0` (`bias_linspace_init_` in `components/init.py` defaults to 3.4 -> 6.0). These two initialization ledgers are not judged; they are documented only for reference.

## General conventions

Every mode returns a dict; field semantics are in "Modes and rules", and unlisted intermediate values are not judged. `eps` is always added **outside the denominator** (`denominator + eps`), never into the numerator or inside the denominator; `logsigmoid(x) = -softplus(-x)` is computed piecewise to avoid overflow; `-inf` entries become `0.0` through `exp`. Defaults are for reference only — judging passes them explicitly, but there are also cases that omit them:

```
DH = len(q[0])            # head dimension, 1..3
S  = len(i)               # sequence length, 1..4
CS = chunk_size           # S must be divisible by CS
logsigmoid(x) = -softplus(-x) = x - log1p(exp(x))        (x < 0)
                              = -log1p(exp(-x))           (x >= 0)
eps   = 1e-6              # added OUTSIDE the denominator
NEG_INF = float("-inf")   # masked entries of log_fg_matrix
```

Three global rules: **(1) `m_new = max(...)` is a log-sum-exp stabilizer, not `logaddexp`, so `fg_act + ig_act` need not equal 1; (2) the lower bounds of `h_denom` and `normalizer` are set by `exp(-m)` / `exp(-max_log_D)`, not simply `|qn|` / `|C.sum|`; (3) the intra-chunk `m_loc` is a chunk-level max, not a global max over the whole sequence.**

## Modes and rules

### 1. stabilized_step: m_state stabilization and one recurrent step

Inputs: `q, k, v` (length-DH lists), `i, f` (scalars), `C_prev` (DHxDH), `n_prev` (DH), `m_prev=0.0`, `eps=1e-6`. Returns `{"log_fg_act", "m_new", "fg_act", "ig_act", "C_new", "n_new", "h_num", "qn", "max_val", "h_denom", "h"}`.

```
log_fg_act = logsigmoid(f)
m_new      = max(log_fg_act + m_prev, i)
fg_act     = exp(log_fg_act + m_prev - m_new)
ig_act     = exp(i - m_new)
k_scaled   = k / sqrt(DH)
C_new[a][b]= fg_act * C_prev[a][b] + ig_act * k_scaled[a] * v[b]
n_new[a]   = fg_act * n_prev[a] + ig_act * k_scaled[a]
h_num[b]   = sum_a q[a] * C_new[a][b]
qn         = sum_a q[a] * n_new[a]
max_val    = exp(-m_new)
h_denom    = max(abs(qn), max_val) + eps
h[b]       = h_num[b] / h_denom
```

- `m_new` is a **per-head scalar** log-sum-exp stabilizer: `max` takes the larger of the two branches, and on a tie it takes the first one (`log_fg_act + m_prev`).
- `C_new` is a `DH x DH` matrix memory: the outer product in `ig_act * (k_scaled @ v.T)` is `k_scaled[a] * v[b]`, never transposed.
- `h_num` is taken **by column** as `q.T @ C_new` (`h_num[b] = sum_a q[a] * C_new[a][b]`), not by row.
- `h_denom = max(|qn|, exp(-m_new)) + eps`: neither `abs`, nor the `exp(-m_new)` lower bound, nor the `eps` placed outside `max(...)` may be dropped.
- `C_prev / n_prev / m_prev` are all explicit inputs; `m_prev` and `eps` have defaults and judging also covers calls that omit them.

### 2. parallel_decay: cumsum-difference D matrix and rowwise stabilization

Inputs: `q, k, v` (S x DH), `i, f` (length S), `stabilize_rowwise=True`, `eps=1e-6`. Returns `{"log_fgates_cumsum", "log_fg_matrix", "log_D_matrix", "max_log_D", "D_matrix", "C_matrix", "normalizer", "h_tilde"}`.

```
log_fg[t]          = logsigmoid(f[t])
cumsum             = [0.0] + prefix_sum(log_fg)          # len S+1, cumsum[0] = 0
log_fg_matrix[t][j]= cumsum[t+1] - cumsum[j+1]   if j <= t else -inf
log_D[t][j]        = log_fg_matrix[t][j] + i[j]          # i broadcast by column
max_log_D[t]       = max_j log_D[t][j]   (rowwise=True)
                   = max_{t,j} log_D[t][j] (rowwise=False, same value for all t)
D[t][j]            = exp(log_D[t][j] - max_log_D[t])      # -inf -> 0
keys_scaled        = k / sqrt(DH)
C[t][j]            = (sum_d q[t][d] * keys_scaled[j][d]) * D[t][j]
normalizer[t]      = max(abs(sum_j C[t][j]), exp(-max_log_D[t]))
h_tilde[t][d]      = sum_j (C[t][j] / (normalizer[t] + eps)) * v[j][d]
```

- **A leading zero prepended to the cumsum plus the `[1:,1:]` index offset** is the classic trap: `log_fg_matrix[t][j] = cumsum[t+1] - cumsum[j+1]` (excluding the forget gate at j itself), with `tril` making `j > t` entries `-inf`.
- `igate_preact.T` **broadcasts by column**: column j adds `i[j]`, not `i[t]`.
- The `-inf` entries for `j > t` in `D_matrix` become `0.0` through `exp`, never 1.
- `stabilize_rowwise=True` takes a max per row; `False` takes one global max over the whole matrix and broadcasts it to all rows. With `eps > 0` the two paths are no longer invariant under per-row rescaling for `D / C / normalizer / h_tilde`.
- The `normalizer` lower bound is `exp(-max_log_D[t])`; neither `abs(sum)` nor the bound may be dropped, and `eps` is again outside the denominator.
- `h_tilde` is `C_normalized @ v`, aggregated over j, **not** over `v[t]`.

### 3. chunk_intra_gates: intra-chunk loggates / m_loc / kv / ksum

Inputs: `q, k, v` (S x DH), `i, f` (length S), `chunk_size=2` (`S % chunk_size == 0`). Returns `{"log_fgates_acc", "loggates", "m_loc", "kv", "ksum"}`.

```
NS                 = S // CS                              # exact division, no remainder
log_fgates_acc[n]  = cumsum over the n-th chunk only     # restarts per chunk
base[t]            = i[t] - log_fgates_acc[t] + log_fgates_acc[-1]
m_loc[n]           = max_t base[t]                       # chunk-level max
loggates[t]        = base[t] - m_loc[n]
e[t]               = exp(loggates[t])
kv[n][a][b]        = sum_t k[n,t,a] * v[n,t,b] * e[t]
ksum[n][a]         = sum_t k[n,t,a] * e[t]
```

- `log_fgates_acc` is an **inclusive cumsum within each chunk**, restarting from 0 for every chunk, not a whole-sequence cumsum.
- The `+ log_fgates_acc[-1]` in `base` cancels together with `m_loc`, so it **only affects the returned `m_loc`**, not `loggates`; it must not be dropped.
- `m_loc` is a **chunk-level** scalar, a max over the `CS` dimension, one per chunk — a different level from the per-token `m_new` in mode 1.
- `kv[n]` is `DH x DH`: `k.T @ (v * exp(loggates))`; `ksum[n]` is `DH`, summed over the `CS` dimension. Both use `k` (not `q`, and with no extra scaling).
- This problem **judges only these 5 intra-chunk intermediates**: no inter-chunk `C / n / m` recurrence, no `intra + inter` composition, and no identity with the recurrent / parallel forms.
- Official `q` is viewed and divided by `sqrt(DH)` in this prefix, but none of the 5 returned values depends on `q`, so the implementation does not use `q`.

## Input and output examples

```
solve(mode="stabilized_step", q=[1.0, 0.0], k=[0.0, 0.0], v=[1.0, 2.0],
      i=0.0, f=0.0, C_prev=[[1.0, 2.0], [3.0, 4.0]], n_prev=[0.5, 0.25])
# {"m_new": 0.0, "fg_act": 0.5, "ig_act": 1.0,
#  "C_new": [[0.5, 1.0], [1.5, 2.0]], "h_num": [0.5, 1.0],
#  "qn": 0.25, "max_val": 1.0, "h_denom": 1.000001,
#  "h": [0.4999995, 0.999999]}

solve(mode="parallel_decay", q=[[0.25], [1.0]], k=[[1.0], [1.0]],
      v=[[2.0], [3.0]], i=[0.0, 0.0], f=[0.0, 0.0])
# {"log_fgates_cumsum": [0.0, -0.6931471805599453, -1.3862943611198906],
#  "D_matrix": [[1.0, 0.0], [0.5, 1.0]],
#  "C_matrix": [[0.25, 0.0], [0.5, 1.0]], "normalizer": [1.0, 1.5],
#  "h_tilde": [[0.4999995], [2.6666649]]}

solve(mode="chunk_intra_gates", q=[[1.0], [1.0], [1.0], [1.0]],
      k=[[1.0], [2.0], [3.0], [4.0]], v=[[1.0], [1.0], [1.0], [1.0]],
      i=[0.0, 0.0, 0.0, 0.0], f=[0.0, 0.0, 0.0, 0.0])
# {"loggates": [[-0.6931471805599453, 0.0], [-0.6931471805599453, 0.0]],
#  "m_loc": [0.0, 0.0], "kv": [[[2.5]], [[5.5]]],
#  "ksum": [[2.5], [5.5]]}
```

## Out of scope

`mLSTMCell` `nn.Linear` projections (igate/fgate weights), `MultiHeadLayerNorm` and multi-head projections, real forward passes and sampling, Triton/CUDA kernels, the inter-chunk `C/n/m` state and `intra + inter` composition of `chunkwise_simple`, identity among the three forms (already covered by `la-gla-chunked`), delta-rule matrix updates (already covered by `la-gated-deltanet-recurrent`), ZOH scalar SSM (already covered by `la-mamba-selective-scan`), WKV vector recurrence (already covered by `la-rwkv-channel-decay`), and the random / linspace initialization of `bias_linspace_init_` / `small_init_init_` / `wang_init_`. Judging covers only the three pure numeric ledgers above.
