# xLSTM mLSTM 的 stabilized 指数门、D 矩阵与 chunk 内门控账

## 背景

`NX-AI/xlstm`（`main`）的 `xlstm/blocks/mlstm/backends.py` 用纯 torch 给出了 mLSTM 的三种等价形式：parallel（并行）、recurrent step（单步递推）、chunkwise（分块）。三者都不含真权重——`q / k / v / i / f` 全部作为入参传入，非常适合抽成**无状态判题**。本题把其中三段独有的中间量抽出来，实现单入口 `solve(mode, **kw)`，共 3 个 mode；所有输入均为小规模数值列表（`DH ≤ 3`、`S ≤ 4`、`chunk_size` 整除 `S`），纯 Python 数值账，不 import `xlstm`、不跑真实模型。

官方 `main` 分支的 `backends.py` 共 265 行；`cell.py` 里 `mLSTMCell` 的状态形状为 `C:(B,NH,DH,DH)` / `n:(B,NH,DH,1)` / `m:(B,NH,1,1)`，fgate bias 用 `start=3.0, end=6.0`（`components/init.py` 的 `bias_linspace_init_` 默认 3.4→6.0）。这两处初始化账本题不判，只作口径说明。

## 通用约定

所有 mode 返回 dict，字段语义见「模式与规则」，未列出的中间量不判。`eps` 一律**加在分母外**（`denominator + eps`），不是加在分子或分母内部；`logsigmoid(x) = -softplus(-x)` 分段计算避免溢出；`-inf` 项经 `exp` 变 `0.0`。默认值仅作口径说明，判题会显式传入，但也有「不传默认参数」的用例：

```
DH = len(q[0])            # head dimension, 1..3
S  = len(i)               # sequence length, 1..4
CS = chunk_size           # S must be divisible by CS
logsigmoid(x) = -softplus(-x) = x - log1p(exp(x))        (x < 0)
                              = -log1p(exp(-x))           (x >= 0)
eps   = 1e-6              # added OUTSIDE the denominator
NEG_INF = float("-inf")   # masked entries of log_fg_matrix
```

三点全局口径：**(1) `m_new = max(...)` 是 log-sum-exp 稳定器，不是 `logaddexp`，因此 `fg_act + ig_act` 不必等于 1；(2) `h_denom` 与 `normalizer` 的下界都由 `exp(-m)` / `exp(-max_log_D)` 决定，不是简单的 `|qn|` / `|C.sum|`；(3) chunk 内的 `m_loc` 是 chunk 级 max，不是全序列全局 max。**

## 模式与规则

### 1. stabilized_step：m_state 稳定化与单步递推

输入 `q, k, v`（长度 DH 列表）、`i, f`（标量）、`C_prev`（DH×DH）、`n_prev`（DH）、`m_prev=0.0`、`eps=1e-6`。返回 `{"log_fg_act", "m_new", "fg_act", "ig_act", "C_new", "n_new", "h_num", "qn", "max_val", "h_denom", "h"}`。

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

- `m_new` 是**逐头标量** log-sum-exp 稳定器：`max` 取两个分支的较大者，相等时取第一个（`log_fg_act + m_prev`）。
- `C_new` 是 `DH×DH` 矩阵记忆：`ig_act * (k_scaled @ v.T)` 的外积方向是 `k_scaled[a] * v[b]`，不能转置。
- `h_num` 按**列**取 `q.T @ C_new`（`h_num[b] = Σ_a q[a]·C_new[a][b]`），不是按行。
- `h_denom = max(|qn|, exp(-m_new)) + eps`：`abs` 不可省，下界 `exp(-m_new)` 不可省，`eps` 加在 `max(...)` 之外。
- `C_prev / n_prev / m_prev` 全作显式入参；`m_prev` 与 `eps` 有默认值，判题也覆盖不传它们的调用。

### 2. parallel_decay：cumsum 差分构造 D 矩阵与 rowwise 稳定化

输入 `q, k, v`（S×DH）、`i, f`（长度 S）、`stabilize_rowwise=True`、`eps=1e-6`。返回 `{"log_fgates_cumsum", "log_fg_matrix", "log_D_matrix", "max_log_D", "D_matrix", "C_matrix", "normalizer", "h_tilde"}`。

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

- **cumsum 首行补 0 + `[1:,1:]` 下标偏移**是易错点：`log_fg_matrix[t][j] = cumsum[t+1] - cumsum[j+1]`（不含 j 自身的 log forget gate），配合 `tril` 使 `j > t` 为 `-inf`。
- `igate_preact.T` 是**按列广播**：第 j 列加 `i[j]`，不是加 `i[t]`。
- `D_matrix` 里 `j > t` 的 `-inf` 项经 `exp` 变 `0.0`，不能当成 1。
- `stabilize_rowwise=True` 逐行取 max；`False` 对全矩阵取一个全局 max 后广播到所有行。两条路径的 `D / C / normalizer / h_tilde` 在 `eps > 0` 时不再逐行缩放不变。
- `normalizer` 的下界是 `exp(-max_log_D[t])`，`abs(sum)` 与下界都不可省；`eps` 同样加在分母外。
- `h_tilde` 是 `C_normalized @ v`，按 j 聚合，**不是**用 `v[t]`。

### 3. chunk_intra_gates：chunk 内的 loggates / m_loc / kv / ksum

输入 `q, k, v`（S×DH）、`i, f`（长度 S）、`chunk_size=2`（`S % chunk_size == 0`）。返回 `{"log_fgates_acc", "loggates", "m_loc", "kv", "ksum"}`。

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

- `log_fgates_acc` 是**段内含入 cumsum**，每个 chunk 从 0 重新起算，不是全序列 cumsum。
- `base` 里的 `+ log_fgates_acc[-1]` 会与 `m_loc` 一起被减掉，**只影响返回的 `m_loc`**，不影响 `loggates`；因此不能省。
- `m_loc` 是对 `CS` 维取 max 的**chunk 级**标量，每个 chunk 一个，与 mode 1 的逐 token `m_new` 不同层级。
- `kv[n]` 是 `DH×DH`：`k.T @ (v * exp(loggates))`；`ksum[n]` 是 `DH`，沿 `CS` 维求和。二者都用 `k`（不是 `q`，也不额外缩放）。
- 本题**只判 chunk 内这 5 个中间量**，不判 chunk 间 `C / n / m` 递推、不判 `intra + inter` 合成、不判与 recurrent / parallel 的恒等。
- `q` 在官方这段前被 `view` 并除以 `sqrt(DH)`，但 5 个返回量都不依赖 `q`，故实现不使用 `q`。

## 输入与返回示例

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

## 不进判题域

`mLSTMCell` 的 `nn.Linear` 投影（igate/fgate 权重）、`MultiHeadLayerNorm` 与多头投影、真模型前向与采样、Triton/CUDA kernel、`chunkwise_simple` 的 chunk 间 `C/n/m` 状态与 `intra + inter` 合成、三形式之间的恒等性（已由 `la-gla-chunked` 覆盖）、delta rule 矩阵更新（已由 `la-gated-deltanet-recurrent` 覆盖）、ZOH 标量 SSM（已由 `la-mamba-selective-scan` 覆盖）、WKV 向量递推（已由 `la-rwkv-channel-decay` 覆盖）、`bias_linspace_init_` / `small_init_init_` / `wang_init_` 的随机或 linspace 初始化。判题只考上述三段纯数值账。
