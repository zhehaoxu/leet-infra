# torchaudio MVDR 波束形成账：PSD 复数外积与 mask 归一、Tikhonov 只改实部对角、Souden 的复数迹归一、RTF 的 denominator.real、rtf_power 的 n_iter 编排、共轭 einsum 应用

## 1. 背景

`torchaudio.functional` 的 MVDR 块（functional.py 1799-2198 行，九个顶层函数）是该文件最后一块未覆盖的大机制簇：从跨通道功率谱 `psd` 出发，经 `_compute_mat_trace` / `_tik_reg` / `_assert_psd_matrices` 三个私有依赖，分别产出 Souden 式权重 `mvdr_weights_souden`、需显式 RTF 的 `mvdr_weights_rtf`、幂迭代估计 `rtf_power`，最后用 `apply_beamforming` 把权重共轭乘回多通道频谱。本题把整条链用纯 Python 复刻，拆成七个可独立判题的 mode。
全库 `mvdr` / `souden` / `rtf_power` / `beamforming` / `diagonal_loading` / `Tikhonov` 零命中；855 / 857 / 858 / 859 / 860 / 861 已吃掉同一文件的频谱后处理、波形算子、NCCF、sinc 重采样、Griffin-Lim 与响度门控，本题与它们的函数集合不重叠。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  psd                    (line 1799)
  _compute_mat_trace     (line 1844)
  _tik_reg               (line 1865)
  _assert_psd_matrices   (line 1886)
  mvdr_weights_souden    (line 1913)
  mvdr_weights_rtf       (line 1976)
  rtf_evd                (line 2061, probe only)
  rtf_power              (line 2085)
  apply_beamforming      (line 2148)
```

### 逐字执行环境

用 `ast.get_source_segment` 从 functional.py 逐字抽出上述九个定义，`exec` 进同一个命名空间，因此 `mvdr_weights_souden` / `rtf_power` 解析到的是真实的 `_assert_psd_matrices` / `_tik_reg` / `_compute_mat_trace`，官方函数体一字未改。复数跨越边界一律用 `[re, im]` 嵌套列表，`C()` 建 complex128 张量、`cx()` 转回列表。`rtf_evd` 依赖 `torch.linalg.eigh`，特征向量的符号/相位由 LAPACK 实现决定，**不进入判题域**，只用探针确认它取 `v[..., -1]`。

## 2. 口径约定

### 输入与输出

所有 mode 走 `solve(mode, **kw)` 单一入口；纯 Python，无 torch / torchaudio / numpy / scipy，无全局 RNG。复数一律 `[re, im]` 二元组，实数用 Python 数；剥除官方的前导 `...` 批次维，判题域固定 `(freq, channel[, channel])` 与 `(channel, freq, time)`。每个 mode 除主结果外还返回中间账（ledger）字段，判题逐项核对。
```text
mode       primary args                         defaults
psd        specgram, mask=None,                normalize=True, eps=1e-10
trace      mat, dim1=-1, dim2=-2
tik_reg    mat,                                reg=1e-7, eps=1e-8
souden     psd_s, psd_n, reference_channel,    diagonal_loading=True,
                                               diag_eps=1e-7, eps=1e-8
rtf        rtf, psd_n, reference_channel=None, diagonal_loading=True,
                                               diag_eps=1e-7, eps=1e-8
rtf_power  psd_s, psd_n, reference_channel,    n_iter=3,
                                               diagonal_loading=True, diag_eps=1e-7
beamform   weights, specgram
```

### 非法输入

校验顺序本身是契约：`_need_cx` 先查维数与矩形叶子，再查方阵/形状一致，最后才 `pack` 转换；`reference_channel` 只允许 int 或非 bool 的 one-hot 列表，`bool` 按官方 `torch.jit.isinstance` 口径抛 `TypeError('Expected "int" or "Tensor" for reference_channel')`。非方阵 psd_n 在 `rtf` 中必然先撞 `rtf.shape == psd_n.shape[:-1]` 一关，专用的方阵检查在该路径不可达。

## 3. mode 1：psd（跨通道功率谱）

`specgram` 是 `(C, F, T)` 复数谱；先 `transpose(-3,-2)` 成 `(F, C, T)`，再按官方 einsum `...ct,...et->...tce` 做 `t * t.conj()` 的复数外积（注意是右操作数共轭）。`mask` 是实数 `(F, T)`，形状不符抛 `The dimensions of mask except the channel dimension should be the same as specgram`；`normalize=True` 时 `mask / (mask.sum(-1, keepdim=True) + eps)`，随后 `mask[..., None, None]` 广播乘进外积，最后在 `dim=-3` 的时间轴求和。返回 `psd` / `shape` / `mask_norm` / `sum_dim=-3`。

## 4. mode 2：trace（复数矩阵迹）

`_compute_mat_trace` 先要求 `ndim >= 2` 与 `shape[dim1] == shape[dim2]`，再 `torch.diagonal(input, 0, dim1, dim2).sum(-1)`；对判题域的 `(F, C, C)` 方阵，两种 dim 次序给出同一条对角。返回 `trace`（复数）/ `diag` / `shape`。

## 5. mode 3：tik_reg（Tikhonov 对角加载）

★ 官方 docstring 明写「only modifying real part」：`epsilon = _compute_mat_trace(mat).real[..., None, None] * reg + eps`，再 `mat + epsilon * eye`。只有对角元的实部变化，非对角元与对角元虚部原样保留；`eye` 与 `mat` 同 dtype/device。返回 `out` / `shape` / `trace_real` / `epsilon` / `eye` / `channels` / `reg` / `eps`。

## 6. mode 4：souden（Souden MVDR 权重）

`_assert_psd_matrices` 校验后，若 `diagonal_loading` 则先对 psd_n 做 `_tik_reg(reg=diag_eps)`；`numerator = torch.linalg.solve(psd_n, psd_s)`，★ 归一除的是 `_compute_mat_trace(numerator)` 的**复数迹**（没有 `.real`），`ws = numerator / (trace[..., None, None] + eps)`。int 参考通道取 `ws[..., :, rc]` 一列，one-hot 张量走 `einsum("...c,...c->...", [ws, rc[..., None, None, :]])`。返回 `weights` / `shape` / `branch` / `psd_n_reg` / `numerator` / `trace_numerator` / `ws`。纯 Python 侧用带部分主元的复数 LU + 回代复刻 `linalg.solve`，奇异矩阵抛 `singular matrix in linalg.solve`。

## 7. mode 5：rtf（显式 RTF 的 MVDR 权重）

`numerator = solve(psd_n, rtf.unsqueeze(-1)).squeeze(-1)`；`denominator = einsum("...d,...d->...", [rtf.conj(), numerator])` 是 `v^H @ numerator`（共轭的是 v，不是 numerator）；★ 归一除的是 `denominator.real.unsqueeze(-1) + eps`，复数分母的虚部被丢弃。若 `reference_channel` 非 None，再乘 `scale`：int 分支 `rtf[..., rc].conj()`，one-hot 分支 `einsum("...c,...c->...", [rtf.conj(), rc[..., None, :]])`；`None` 时 `branch="none"`、`scale=None`。返回 `weights` / `shape` / `branch` / `psd_n_reg` / `numerator` / `denominator` / `denominator_real` / `weights_unnorm` / `scale`。

## 8. mode 6：rtf_power（幂迭代 RTF 估计）

`n_iter <= 0` 抛 `The number of iteration must be greater than 0.`。加载后 `phi = solve(psd_n, psd_s)`（官方注释视作第一次迭代），按参考通道取 `phi[..., rc]` 或 einsum 得 `r0`，再 `unsqueeze(-1)`。★ 编排口径：`n_iter >= 2` 时先跑 `n_iter - 2` 次 `phi @ cur`，最后乘一次 `psd_s @ cur`（`tail="psd_s"` 且 `n_loop = n_iter - 2`）；`n_iter == 1` 时只做一次 `psd_n @ cur`（`tail="psd_n"`、`n_loop=0`）——注释解释这是 `psd_n @ phi @ u = psd_s @ u`。返回 `rtf` / `shape` / `branch` / `psd_n_reg` / `phi` / `rtf_init` / `n_loop` / `tail` / `n_iter`。

## 9. mode 7：beamform（共轭 einsum 应用）

`apply_beamforming` 只做 `einsum("...fc,...cft->...ft", [w.conj(), specgram])`：权重先共轭，再沿通道维与 `(C, F, T)` 谱逐频逐帧点乘，输出 `(F, T)`。剥除批次维后形状校验退化为通道数 C 与频率数 F 必须一致。返回 `out` / `shape` / `weights_conj`。

## 10. 官方 quirk 清单

`psd` 的 docstring 写 eps 默认 `1e-15`、签名却是 `1e-10`；`_tik_reg` 的 docstring 写 reg 默认 `1e-8`、签名却是 `1e-7`（本题按签名）。`mvdr_weights_rtf` 的报错里有官方拼写 `withou`；`mvdr_weights_souden` 的 docstring 里有 `martrix`。`_assert_psd_matrices` 与 `rtf` 的复数 dtype 检查在纯 Python 域退化为 `[re, im]` 叶子校验。`rtf_evd` 用 `v[..., -1]` 但特征向量相位不可移植，故整段排除。

## 11. 示例与落地

官方逐字环境 12 段探针 241/241、三层对拍 134334 项 / 18 组配置 ALL GREEN、判题 73/73、变异 130 项剔除 2 个真等价后 128/128 全 CAUGHT。落地 11 文件（四份双语文档 + meta + solution/starter/test_cases + variants/pytorch 三件套），`collections.json` 注册 `audio`，短护栏与全量护栏分别独立 `--basetemp` 运行。详见 `notes.md`。
