# 官方机制与易错点笔记

## 官方来源与版本

- 仓库：`NX-AI/xlstm`，默认分支 `main`。
- 用到的文件：`xlstm/blocks/slstm/src/vanilla/slstm.py`（`slstm_forward_pointwise`）、`xlstm/blocks/slstm/src/vanilla/lstm.py`（普通 LSTM 对照）、`xlstm/blocks/slstm/cell.py`（`sLSTMCellConfig`、`reset_parameters`、`_zero_state`）、`xlstm/blocks/slstm/src/cuda/slstm_pointwise.cu` 与 `slstm_forward.cu`（kernel 语义对照）、`xlstm/components/init.py`（`bias_linspace_init_`）。
- 回源方式：GitHub API `git/trees/main?recursive=1` + `contents?ref=main` + `base64.b64decode`。
- 官方对拍脚本用 `.venv-eval/bin/python`（torch 2.14.1）**原样 exec** 官方 `vanilla/slstm.py`，用 torch 张量构造 `[B=1, 4H]` 与 `states[4, 1, H]`，再与自写纯 Python solve 逐字段比对。

## 与 mLSTM（837 题）的划界

837 题判的是 `xlstm/blocks/mlstm/backends.py` 的 **mLSTM**：**矩阵记忆** `C`（DH×DH）、`sqrt(DH)` 缩放、`chunkwise_simple` 的 chunk 内门控。本题判的是 `xlstm/blocks/slstm/src/vanilla/slstm.py` 的 **sLSTM**：

- 状态维度不同：mLSTM 是矩阵 `C` + 向量 `n` + 标量 `m`；sLSTM 是**每通道一个标量** `c` + 标量 `n` + 标量 `m`，共 4 态 `[y, c, n, m]`。
- 门控不同：mLSTM 用 `logaddexp` 风格的 `max` 稳定化（`m_new = max(log_fg_act + m_state, igate_preact)`）；sLSTM 用 `m + logsigmoid(fraw)` 加 `min(·, 1)` 硬截断。
- 文件不同、判题域不重叠。本题**不判**三形式恒等、不判矩阵记忆等价性、不判 chunkwise。

## sLSTM 点式前向的五个坑

1. **`torch.all(n == 0.0)` 是整帧分支**。官方 vanilla 实现里 `n` 是整个 `[B, H]` 张量，`torch.all` 作用在整个张量上，所以要么整帧走 `mnew = iraw`，要么整帧走 `mnew = max(iraw, logfplusm)`。**逐元素判断是错的**——CUDA kernel 侧才是逐元素的 `eq_zero_g(n_cur)`。这是 sLSTM 与 mLSTM stabilized gating 最刺眼的差别，题面明确按 vanilla 的整帧语义。
2. **`logfplusm = m + logsigmoid(fraw)`**。`fraw` 先过 `logsigmoid`（不是 `sigmoid`），再**加上**上一帧的 `m`。漏掉 `m_prev` 会让稳定器完全失效。
3. **`igate` / `fgate` 都要 `min(·, 1)` 硬截断**。`torch.minimum(torch.exp(...), torch.ones_like(...))` 是显式上界，即使 `exp` 结果超过 1 也会被压回 1。
4. **`tanh` 只作用在 `zraw` 上**：`cnew = fgate * c + igate * torch.tanh(zraw)`。如果误写成 `tanh(cnew)` 或 `tanh(zraw)` 之外的位置，结果完全不同。
5. **`ynew = ogate * cnew / nnew`**，分母是 **normalizer state** `nnew`，不是 `cnew`。sLSTM 用 `n` 归一化标量记忆，这是它相对普通 LSTM 的关键差异。

## `max` 稳定器与 bias 初始化的坑

- `m_new = max(iraw, logfplusm)` 保证 `igate = exp(iraw - m_new) <= 1`、`fgate = exp(logfplusm - m_new) <= 1`，两个门天然不超过 1，所以 `min(·, 1)` 只在 `zero_n_branch` 分支下对 `fgate` 真正起作用（此时 `m_new = iraw`，`fgate` 可能远大于 1）。这一点在写 mutation 时要留意：`igate` 的截断是**永不生效**的等价格。
- `powerlaw_blockdependent` 初始化：指数是 `0.3 + 1.3 * ratio`，`ratio = _block_idx / (_num_blocks - 1)`（`_num_blocks == 1` 时为 `0.0`），base 是 `arange(head_dim) / (head_dim - 1)`，**负号在整个括号外**（`-( -5 + 12 * base^exp )`），所以 base 为 0 时结果是 `+5`。
- `head_dim == 1` 时 `arange(1)/(1-1)` 是 `0/0`，`powerlaw` 模式下 bias 为 `NaN`；`small_init` 模式下 `torch.linspace(3.0, 6.0, 1)` 给 `[3.0]`。
- `small_init` 的 `f` 门 bias 是 `torch.linspace(3.0, 6.0, head_dim)`，**含端点**；其余门 bias 全零。

## 验证方式与判题设计

- 官方回源对拍：`pointwise_step` 600 组、`max_stabilizer` 600 组、`bias_powerlaw` 全部 `head_dim × num_blocks × block_idx × mode` 组合，**全部一致**。
- 判题用例：3 个 `check_*`，各含独立纯 Python oracle、手算锚点与网格对拍。
- 手算锚点钉住：`torch.all` 的整帧分支、`logfplusm` 里的 `m_prev`、`minimum(·,1)` 硬上界、`tanh` 只作用在 `zraw`、`ynew` 的分母是 `nnew`、`n_new == 0` 时 `y_new` 为 `NaN`、powerlaw 的指数与括号外负号、`head_dim == 1` 的 `NaN`、`small_init` 的 `linspace(3.0, 6.0)`。
- mutation testing：42 个单点变异，全部 CAUGHT，且**正确解双模式自检先行确认 ACCEPTED**。

## 对拍与 mutation 过程中踩到的坑（可复用经验）

1. **对拍脚本的比较器要支持嵌套列表**。第一版 `cmp_value` 只处理标量，`raw_gates` 这种 4×H 的二维结构直接报 `TypeError: float() argument must be ... not 'list'`。
2. **期望值必须用 `torch.nn.functional.logsigmoid`，不能写 `torch.log(torch.sigmoid(x))`**。后者先 sigmoid 再 log，在 x 接近 0 时抵消误差约 2e-8，与官方实现差一个量级。官方源码用的是 `from torch.nn.functional import logsigmoid`。
3. **对拍里的 `torch.tensor(x)` 默认 float32**，指数运算会带 1e-7 级相对误差。必须显式 `dtype=torch.float64`。
4. **变异要区分「真等价」与「覆盖缺口」**。首轮 41 变异有 5 个存活，诊断后确认全是真等价：
   - `raw = wx + ry + b` 里交换 `wx`/`ry` 是加法交换律；
   - `igate` 的 `min(·,1)` 永不生效（`m_new >= iraw` 恒成立）；
   - `gate_le_one` 恒为 True（两个门都被截到 `<= 1`）；
   - `_ = y_prev` 是 no-op。
   按既定原则，真等价变异必须剔除并换成可测形态（分别换成「bias 减号」「逐元素分支」「严格小于」「`m_prev` 取错 state」）。
5. **窗口太窄的变异等于准等价**。把 `if gf > 1.0` 改成 `if gf > 1.0000001`，网格里几乎撞不上，仍然存活。改成「截断值改成 0.999999」才真正可测。
6. **手算锚点要自己复核**。本轮 `max_stabilizer` 锚点 1 漏了 `fgate` 的 `min(·,1)` 截断（通道 1 原始值 4.154 应为 1.0），`pointwise_step` 锚点 3 把 `c_new = fgate*c + igate*tanh(zraw)` 误算成 1.5（正确是 0.5），都被自检立刻抓出。

## 不进判题域的边界

`R` 矩阵与状态的 GEMM 及其 `reshape`/`transpose` 布局链、CUDA kernel 的 launch 配置与 `round_to_power2`、`SLSTM_FORWARD_CLIPVAL`、autograd 反向与梯度裁剪、参数随机初始化（`zeros`/`standard` 权重、`small_init_init_` 的正态采样）、`vanilla/lstm.py` 的普通 LSTM 变体、多头拆分与 `num_gates != 4` 的配置，全部不进判题域。本题只保留三段确定性的纯数值账。
