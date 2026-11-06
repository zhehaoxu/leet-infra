# 解题笔记：xLSTM mLSTM 的 stabilized 门、D 矩阵与 chunk 内门控账

## 回源口径

- 仓库 `NX-AI/xlstm`，默认分支 `main`（122 blobs，truncated=False，pushed 2026-10-24）。通过 GitHub API `git/trees/main?recursive=1` 取树，再用 `contents?ref=main` + base64 下载到 `/tmp/xlstm_src/`；`raw.githubusercontent.com` 通道在 833 期已证实不稳，不用。
- 核心源码 `xlstm/blocks/mlstm/backends.py`（265 行）：`parallel_stabilized_simple`（L9–90）、`recurrent_step_stabilized_simple`（L93–146）、`chunkwise_simple`（L149–263）。另参考 `cell.py`（141 行，mLSTMCell 状态形状 `C:(B,NH,DH,DH)` / `n:(B,NH,DH,1)` / `m:(B,NH,1,1)`，fgate bias `start=3.0, end=6.0`）与 `components/init.py`（`bias_linspace_init_` 默认 3.4→6.0），二者本题不判，只作口径说明。
- 官方对拍脚本 `/tmp/verify837.py` 用 `ast.get_source_segment` 原样 exec 官方三个函数（`chunkwise_simple` 截到 `C = torch.zeros(...)` 之前的前段并追加 return 暴露 5 个中间量），以 `.venv-eval/bin/python` 跑 **1800 组**（step / parallel / chunk 各 600）全一致；step 比 `h / C_new / n_new / m_new / log_fg_act / fg_act / ig_act / qn / max_val / h_denom`，parallel 比 `h_tilde`，chunk 比 `log_fgates_acc / loggates / m_loc / kv / ksum`。官方函数吃 5 维 torch tensor，对拍脚本负责形状适配。

## 机制要点

- `stabilized_step`：`m_new = max(log_fg_act + m_prev, i)` 是 log-sum-exp 稳定器，**不是 `logaddexp`**，故 `fg_act + ig_act` 不必等于 1；`C_new` 是 `DH×DH` 矩阵记忆，外积方向为 `k_scaled[a] * v[b]` 不可转置；`h_num` 按列取 `q.T @ C_new`；`h_denom = max(|qn|, exp(-m_new)) + eps`，下界由 `m_new` 决定而不是单纯 `|qn|`，`eps` 加在 `max(...)` 之外。
- `parallel_decay`：`cumsum = [0] + prefix_sum(log_fg)` 后取 `[1:,1:]` 偏移，`log_fg_matrix[t][j] = cumsum[t+1] - cumsum[j+1]`（不含 j 自身的 forget gate），`j > t` 置 `-inf`；`igate_preact.T` 按列广播加 `i[j]` 而非 `i[t]`；`stabilize_rowwise` 决定逐行 max 或全局 max；`D` 的 `-inf` 经 `exp` 变 `0.0`；`normalizer = max(|C.sum|, exp(-max_log_D))`，`h_tilde = (C / (normalizer + eps)) @ v`。
- `chunk_intra_gates`：`log_fgates_acc` 是**段内含入 cumsum**，每个 chunk 从 0 重启；`base = i - acc + acc[-1]` 中 `+acc[-1]` 与 `m_loc` 相消，只影响返回的 `m_loc`，不可省；`m_loc` 是 chunk 级 max（对 CS 维），与 mode 1 的逐 token `m_new` 不同层级；`kv = k.T @ (v * exp(loggates))`、`ksum = (k * exp(loggates)).sum(CS)`，二者只用 `k` 不缩放，`q` 在这 5 个返回量里完全不参与；`S` 必须被 `chunk_size` 整除。
- 全局口径：`logsigmoid(x) = -softplus(-x)` 分段避免溢出；`eps` 一律加在分母外；所有输入 q/k/v/i/f 全作显式入参，纯 Python 数值账，不 import `xlstm`、不跑真实模型。

## 测试设计

- 3 个 `check_*`（`check_stabilized_step` / `check_parallel_decay` / `check_chunk_intra_gates`），每个都有独立纯 Python oracle（不 import solution）+ 网格对拍（160 + 160 + 160）+ 手算锚点（4 + 3 + 4）。
- 手算锚点刻意钉死所有易错点：`m_new = max` 两分支（含相等时取前者）、`sqrt(DH)` 缩放、外积方向、`h_num` 按列、`h_denom` 的 `max / abs / eps` 与下界、cumsum 首 0 与 `[1:,1:]` 偏移、`-inf` 变 0、igate 按列广播、`stabilize_rowwise` 两路、`normalizer` 下界 `exp(-max_log_D)`、chunk 内 cumsum 重启、`+acc[-1]` 只改 `m_loc`、chunk 级 `m_loc`（两个 chunk 分别为 0 与 10）、`kv[a][b]` 方向、`ksum` 沿 CS 维求和、`chunk_size=1`。
- 四个默认值（`m_prev` / `eps` / `stabilize_rowwise` / `chunk_size`）都有「不传该参数」的独立锚点，否则改成别的默认值的变异不会被发现。
- 正式 selftest 用 `scripts/add_problem.py --selftest <slug> --mode run|submit` 双模式；判题环境只有 torch + numpy，本题为纯 Python 数值账，无需 torch。

## 变异设计

- `/tmp/mut837.py` 先自检正确解在 `run` / `submit` 双模式均 3/3 ACCEPTED（836 教训：mutation 全绿不能替代正确解自检），再跑 **57 个单点变异，首轮 57/57 CAUGHT, 0 SURVIVED**，锚点计数 0 错。
- 变异覆盖：`m_new` 改 `logaddexp` / 去掉 `m_prev`、`fg_act / ig_act` 互换、`k_scaled` 漏除 `sqrt(DH)`、外积转置、`h_num` 按行、`h_denom` 去 `abs` / 去 `exp(-m_new)` 下界 / `eps` 加在分子、cumsum 首 0 漏掉、`[1:,1:]` 偏移错、`-inf` 变 1、igate 按行广播、`stabilize_rowwise` 默认值反转、`normalizer` 去下界、chunk 内 cumsum 不重启、`+acc[-1]` 删掉、`m_loc` 用全局 max、`kv / ksum` 误用 `q` 或漏 `exp(loggates)`、`S % CS` 余数处理等。
- 首轮全绿的原因是把 836 的教训前置：锚点已覆盖所有默认值与严格/非严格边界，且每个变异都死于断言而非形状崩溃；真等价变异（如 `max` 两分支相等时的写法差异）在写 mut 脚本时剔除。

## 查重划界

- 与 `la-gla-chunked`（逐 key 通道 log 衰减的 chunk 并行 vs 递推恒等）、`la-gated-deltanet-recurrent`（`v' = (v − h·k)·β` delta rule 矩阵更新）、`la-mamba-selective-scan`（ZOH 标量 SSM）、`la-rwkv-channel-decay`（WKV 向量递推）划清：本题**不判三形式恒等、不判 delta rule、不判矩阵记忆等价性**，只取 mLSTM 独有的 stabilized gating / parallel D 矩阵 / chunk intra gates 三段中间量。
- 与 `context-compression` 的 KV 压缩、diffusion 的 scheduler 账、data-tokenization 的分段账均无重叠；`context-compression / SnapKV` 方向因 `kv-snapkv-pooling` 已占而整体排除。
- 不进判题域：`mLSTMCell` 的 `nn.Linear` 投影（igate/fgate 权重）、`MultiHeadLayerNorm` 与多头投影、真模型前向与采样、Triton/CUDA kernel、`chunkwise_simple` 的 chunk 间 `C/n/m` 状态与 `intra + inter` 合成、`bias_linspace_init_` / `small_init_init_` / `wang_init_` 的随机或 linspace 初始化。
