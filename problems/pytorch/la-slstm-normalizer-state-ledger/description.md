# xLSTM sLSTM 的 max 稳定器、normalizer state 与标量记忆点式递推账

## 背景

xLSTM 论文里的 **sLSTM**（scalar LSTM）与 mLSTM 并列：它不用矩阵记忆，而是给每个通道一个标量记忆 `c`，再配一个 **normalizer state** `n` 做分母，并用 **max 稳定器** `m` 把指数门控压进数值安全区。

官方 `xlstm/blocks/slstm/src/vanilla/slstm.py` 的 `slstm_forward_pointwise` 把这一步写得很紧凑：

- `m` 是逐通道的标量稳定器，保证两个门都不超过 1；
- `n` 是 normalizer state，累加写入量，最终作为输出的分母；
- 首帧（`n` 全零）走一条特殊分支。

本题重建这三段**纯数值**账：`max` 稳定器与双门、标量记忆 + normalizer state 的点式递推、以及 `f` 门 bias 的两种初始化。GEMM 布局、CUDA kernel、autograd 反传都不在判题域内——输入一律是已经算好的数值列表。

## 通用约定

- 所有输入都是 Python `float` 组成的列表；不涉及张量、梯度和批处理。
- 浮点比较采用绝对误差 `1e-9` 加相对误差 `1e-9`。
- `head_dim` 记作 `H`，即通道数。
- 返回值统一是 `dict`，字段名见各 mode 说明。
- 下标一律从 0 开始，且**闭区间**。

## 模式与规则

### 1. `max_stabilizer`：max 稳定器与双门

入参：

| 参数 | 含义 |
|---|---|
| `iraw` | 输入门预激活，长度 `H` |
| `fraw` | 遗忘门预激活，长度 `H` |
| `m_prev` | 上一帧的 max 稳定器，长度 `H` |
| `n_prev` | 上一帧的 normalizer state，长度 `H` |

计算步骤：

```
logfplusm[i] = m_prev[i] + logsigmoid(fraw[i])
zero_n_branch = 全部 n_prev[i] == 0          # 整帧判定，不是逐元素
if zero_n_branch:
    m_new[i] = iraw[i]
else:
    m_new[i] = max(iraw[i], logfplusm[i])
igate[i] = min(exp(iraw[i] - m_new[i]), 1)
fgate[i] = min(exp(logfplusm[i] - m_new[i]), 1)
```

返回字段：

- `logfplusm`：长度 `H`；
- `m_new`：长度 `H`；
- `zero_n_branch`：布尔，表示是否走了全零分支；
- `igate` / `fgate`：长度 `H`；
- `gate_le_one`：长度 `H` 的布尔列表，逐通道表示两个门是否都不超过 1。

易错点：

- `logfplusm` 用的是 `logsigmoid(fraw)`，不是 `sigmoid(fraw)`；
- `m_prev` 必须参与 `logfplusm` 的求和；
- **`zero_n_branch` 是整帧判定**：只要有一个通道的 `n_prev` 非零，整帧就走 `max` 分支，所有通道统一处理；
- 两个门都要被 `min(·, 1)` 硬截断；
- `logsigmoid(x)` 定义为 `-softplus(-x)`，即 `x >= 0` 时为 `-log(1 + exp(-x))`，否则为 `x - log(1 + exp(x))`。

### 2. `pointwise_step`：标量记忆 + normalizer state 递推

入参：

| 参数 | 含义 |
|---|---|
| `wx` | 输入侧投影，长度 `4H` |
| `ry` | 循环侧投影，长度 `4H` |
| `bias` | 门偏置，长度 `4H` |
| `states` | 4 组长度 `H` 的列表，依次是 `y` / `c` / `n` / `m` |
| `head_dim` | 通道数 `H` |

计算步骤：

```
raw = wx + ry + bias                                  # 长度 4H
iraw, fraw, zraw, oraw = 按 [i, f, z, o] 每组 H 个拆开
logfplusm / m_new / igate / fgate      # 同 max_stabilizer
ogate[i] = sigmoid(oraw[i])
zval[i]  = tanh(zraw[i])
c_new[i] = fgate[i] * c_prev[i] + igate[i] * zval[i]
n_new[i] = fgate[i] * n_prev[i] + igate[i]
y_new[i] = ogate[i] * c_new[i] / n_new[i]
```

返回字段：

- `raw_gates`：4 组长度 `H` 的列表，依次是 `iraw` / `fraw` / `zraw` / `oraw`；
- `logfplusm` / `m_new` / `igate` / `fgate`：长度 `H`；
- `zero_n_branch`：布尔；
- `ogate` / `zval` / `c_new` / `n_new` / `y_new`：长度 `H`。

易错点：

- `raw` 的门顺序是 `[i, f, z, o]`，每组连续 `H` 个；
- `tanh` **只作用在 `zraw` 上**（`igate * tanh(zraw)`），不是作用在 `c_new` 上；
- `y_new` 的分母是 `n_new`（normalizer state），不是 `c_new`；
- 输出**不做** `tanh` 包裹；
- 当 `n_new[i] == 0` 时 `y_new[i]` 记为 `NaN`（IEEE 754 的 `0/0` 语义）。

### 3. `bias_powerlaw`：`f` 门 bias 初始化

入参：

| 参数 | 含义 |
|---|---|
| `head_dim` | 通道数 `H` |
| `block_idx` | 当前 block 序号，从 0 开始 |
| `num_blocks` | block 总数 |
| `bias_mode` | 可选，`powerlaw`（默认）或 `small_init` |

`powerlaw` 模式：

```
ratio = block_idx / (num_blocks - 1)   if num_blocks > 1 else 0.0
exponent = 0.3 + 1.3 * ratio
bias[d] = -( -5.0 + 12.0 * (d / (H - 1)) ** exponent )      # d = 0 .. H-1
```

`small_init` 模式：

```
bias[d] = linspace(3.0, 6.0, H)[d]      # 含端点
```

返回字段：`bias`（长度 `H`）、`exponent`（浮点）、`ratio`（浮点）。

易错点：

- **负号在整个括号外**：`bias = -( -5 + 12 * base^exponent )`，所以 `base = 0` 时 `bias = 5`；
- `exponent` 随 block 序号增大（`0.3` 到 `1.6`）；
- `num_blocks == 1` 时 `ratio = 0.0`；
- `H == 1` 时 `d / (H - 1)` 是 `0 / 0`，`powerlaw` 模式下 `bias` 为 `NaN`；
- `small_init` 在 `H == 1` 时给 `[3.0]`，`exponent` 记为 `0.0`。

## 输入与返回示例

`max_stabilizer`：

```python
solve("max_stabilizer", iraw=[1.0, -2.0], fraw=[0.5, 0.25],
      m_prev=[0.0, 0.0], n_prev=[0.0, 0.0])
# {
#   "logfplusm":     [-0.4741, -0.5759],
#   "m_new":         [1.0, -2.0],          # 全零分支 -> m_new = iraw
#   "zero_n_branch": True,
#   "igate":         [1.0, 1.0],
#   "fgate":         [0.2290, 1.0],        # 通道 1 原始值 4.154 被截断到 1
#   "gate_le_one":   [True, True],
# }
```

`pointwise_step`：

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

`bias_powerlaw`：

```python
solve("bias_powerlaw", head_dim=4, block_idx=0, num_blocks=1)
# {
#   "bias":     [5.0, -3.6307, -5.6256, -7.0],
#   "exponent": 0.3,
#   "ratio":    0.0,
# }
```

## 不进判题域

- `R` 矩阵与状态之间的 GEMM、`reshape` / `transpose` 布局链；
- CUDA kernel 的 launch 配置、`round_to_power2`、`SLSTM_FORWARD_CLIPVAL`；
- autograd 反向传播与梯度裁剪；
- 参数的随机初始化（`zeros` / `standard` 权重、`small_init_init_` 的正态采样）；
- `vanilla/lstm.py` 的普通 LSTM 变体（`sigmoid` 门、`tanh(c)` 输出）——只作对照；
- 多头拆分与 `num_gates` 取其它值的配置。
