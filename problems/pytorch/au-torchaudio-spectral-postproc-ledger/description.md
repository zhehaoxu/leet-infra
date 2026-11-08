# torchaudio 频谱后处理账：amplitude_to_DB 的 per-batch cutoff 与 packed_channels、DB_to_amplitude 的 power 口径、compute_deltas 的互相关 denom、sliding_window_cmn 的窗口递推与切片笔误、spectrogram 的 frame/window 双归一化与 DFT 帧构造

## 1. 背景

本题重建 `torchaudio.functional` 里五段频谱后处理链的**数值口径**，而不是调用库函数。判题环境没有 `torchaudio`，也不允许依赖 `torch`；所有 mode 必须是纯 Python 的确定性数值账，输入是嵌套列表，输出是同形状的嵌套列表（或题目约定的复合字典）。

### 官方源码位置

回源目标是 `pytorch/audio main::src/torchaudio/functional/functional.py`（2511 行）。本题只取其中七个顶层函数：`spectrogram`、`_get_spec_norms`、`_get_complex_dtype`、`amplitude_to_DB`、`DB_to_amplitude`、`compute_deltas`、`sliding_window_cmn`。其中 `_get_spec_norms` 决定归一化的两个布尔开关，`_get_complex_dtype` 给出实 dtype 到复 dtype 的映射。

### 逐字执行环境

用 `ast.get_source_segment` 从官方文件抽出这七个函数的**逐字源码**（签名、docstring、注释全部保留），注入 `torch`、`math`、`warnings`、`Tensor` 与 typing 名字后 `exec` 到同一个命名空间。官方源码一行不改；驱动层只负责把嵌套列表转成 `float64` 张量、记录 `torch.stft` 的调用参数、把复数结果拆成 `real` / `imag`。所有对拍都在 `float64` 下完成，避免 `float32` 量化干扰。

## 2. 口径约定

### 输入与输出

单入口是 `solve(mode, **kw)`，五个 mode 的必填参数与默认值如下。注意 `deltas` 的填充模式在外部叫 `pad_mode`（避免与分派参数 `mode` 重名），但它对应官方 `compute_deltas` 的第三个参数 `mode`。

```text
solve("db",          x, multiplier, amin, db_multiplier, top_db=None)
solve("db_inverse",  x, ref, power)
solve("deltas",      specgram, win_length=5, pad_mode="replicate")
solve("cmn",         specgram, cmn_window=600, min_cmn_window=100,
                     center=False, norm_vars=False)
solve("spectrogram", waveform, window, n_fft, hop_length, win_length,
                     power=2.0, normalized=False, center=True,
                     pad_mode="reflect", onesided=True, pad=0,
                     return_complex=None)
```

每个 mode 返回一个字典，`out` 是数值结果，`shape` 是结果的形状，其余字段是与机制直接相关的元数据（例如 `cutoffs`、`windows`、`kernel`、`frame_norm`）。判题既比数值也比元数据，所以元数据必须与实际计算口径一致，不能写死。

### 数值精度

全部用 Python `float`（即 IEEE double）。对数用 `math.log10`，DFT 用 `math.cos` / `math.sin` 或等价的 `cmath.exp`，累加用 `math.fsum` 或普通求和均可，只要与官方 `float64` 结果在 `1e-8` 相对容差内一致。不允许使用全局随机源：所有扰动、窗口、序列都显式入参。

## 3. mode 1: db

### 公式

`amplitude_to_DB` 先算 `x_db = multiplier * log10(clamp(x, min=amin))`，再原地减去 `multiplier * db_multiplier`。`multiplier=10` 用于功率谱，`20` 用于幅度谱；`amin` 必须为正，`db_multiplier` 通常是参考值的 `log10`。

```text
x_db = multiplier * log10(max(x, amin)) - multiplier * db_multiplier
packed_channels = shape[-3] if ndim > 2 else 1
num_batches     = prod(shape[:-3]) if ndim > 3 else 1
chunk           = packed_channels * freq * time
cutoff[b]       = max(vals[b * chunk : (b+1) * chunk]) - top_db
out             = max(x_db, cutoff[b])
```

### top_db 与 packed_channels

只有 `top_db` 非 `None` 才做截断。官方把张量 `reshape(-1, packed_channels, F, T)`，其中 `packed_channels = shape[-3] if ndim > 2 else 1`，再对 `(-3,-2,-1)` 求每个 batch 的 `amax`。因此 4D 及以上输入**每个 batch 独立 cutoff**，3D 输入（`packed_channels` 是通道数）与 2D 输入都只有**一个** cutoff。返回值必须给出 `cutoffs`（`top_db=None` 时为 `None`）、`packed_channels`、`num_batches`，且 `cutoffs` 的长度等于 `num_batches`。

### 域约束

`x` 至少二维、矩形、非空且只含数字；`multiplier`、`amin`、`db_multiplier`、`top_db` 必须是数字，`amin > 0`。违反时抛 `ValueError`，消息包含被违反的约束名。

## 4. mode 2: db_inverse

### 公式

`DB_to_amplitude` 是 `ref * (10^(0.1*x))^power`。`power=1` 把 dB 还原成功率谱（配合 `multiplier=10`），`power=0.5` 还原成幅度谱（配合 `multiplier=20`）。因此 `solve("db", ...)` 与 `solve("db_inverse", ...)` 在这两种配对下应能往返复原原始谱。返回值包含 `ref` 与 `power` 原值。

### 域约束

`x` 至少一维（允许 1D 列表），只含数字；`ref`、`power` 必须是数字。

## 5. mode 3: deltas

### 互相关与 denom

`compute_deltas` 要求 `win_length >= 3`，令 `n = (win_length - 1) // 2`。注意**偶数**窗口长度也合法，此时 `n` 与 `win_length // 2` 不同。官方分母写成 `n*(n+1)*(2n+1)/3`，等价于 `2 * sum(i^2 for i in 1..n)`。核是 `arange(-n, n+1)`，`conv1d(groups=channels)` 是**互相关**语义：`d[t] = sum_j kernel[j] * padded[t+j] / denom`，与 docstring 的 `sum_i i*(c[t+i] - c[t-i]) / (2*sum i^2)` 一致。对递增序列，内部位置的 delta 应为正。

```text
n      = (win_length - 1) // 2
denom  = n * (n + 1) * (2 * n + 1) / 3
kernel = [-n, ..., 0, ..., n]
padded = replicate_or_constant_pad(row, n)
d[t]   = sum_j kernel[j] * padded[t + j] / denom
```

### 填充模式

`pad_mode="replicate"` 在左右各复制边界值 `n` 次；`"constant"` 补零。两者在边界帧必然不同，在内部帧相同（只要窗口不触及边界）。输入会先 `reshape(1, -1, T)`，所以所有前导维度被展平成通道，返回时再还原原形状。返回值包含 `n`、`denom`、`kernel`、`mode`、`num_channels`，其中 `mode` 是实际使用的填充模式名。

### 域约束

`win_length` 必须是 `int` 且 `>= 3`；`pad_mode` 只能是 `replicate` / `constant`；`specgram` 至少二维。

## 6. mode 4: cmn

### 窗口调度

`sliding_window_cmn` 把最后两维解释为 `(num_frames, num_feats)`，逐帧维护 `[window_start, window_end)`。非 center 时窗口在左侧，center 时以当前帧为中心；负起点会被夹到 0 并把终点右移；非 center 且 `window_end > t` 时 `window_end = max(t+1, min_cmn_window)`；终点越过总帧数时左端同步回退并夹到 0。

```text
if center:  ws = t - cmn_window // 2 ; we = ws + cmn_window
else:       ws = t - cmn_window      ; we = t + 1
if ws < 0:  we -= ws ; ws = 0
if not center and we > t: we = max(t + 1, min_cmn_window)
if we > num_frames: ws -= we - num_frames ; we = num_frames ; ws = max(ws, 0)
window_frames = we - ws
out[t] = x[t] - cur_sum / window_frames
```

### 递推与 norm_vars

首帧一次性累加 `[ws, we)` 的帧和与平方和；后续帧只在左端增加时移除 `last_window_start` 帧、右端增加时加入 `last_window_end` 帧。`norm_vars=True` 时再乘 `variance^-0.5`，其中 `variance = cur_sumsq/T - (cur_sum/T)^2`；`window_frames == 1` 时该帧直接置零。返回值包含 `windows`（逐帧 `[ws, we]` 列表）、`num_frames`、`num_feats`、`norm_vars`。

### 官方 quirk

官方首帧切片写成 `specgram[:, window_start : window_end - window_start, :]`，上界误用了「窗口长度」而不是 `window_end`。但这段只在 `last_window_start == -1`（即 `t=0`）执行，而第一帧的 `window_start` 必定被夹到 0，所以 `window_end - window_start == window_end`，**该笔误在数值上不可观测**。另一个真实可观测的 quirk：2D 输入且只有一帧时，末尾的 `squeeze(0)` 会把输出从 `(1, F)` 压成 `(F,)`，即降成一维列表。docstring 把输入写成 `(..., time, freq)`、输出写成 `(..., freq, time)`，与源码实际的最后两维含义也不一致。

### 域约束

`cmn_window`、`min_cmn_window` 必须是正整数；`center`、`norm_vars` 必须是 `bool`；`specgram` 至少二维。

## 7. mode 5: spectrogram

### 帧构造与 DFT

先做 `pad` 的 constant 双侧补零；若 `center=True`，再按 `pad_mode` 在两侧各补 `n_fft//2`（reflect 的左段是 `x[p], x[p-1], ...`，右段是 `x[-2], x[-3], ...`）。`win_length < n_fft` 时窗口被居中补零：`left = (n_fft - win_length)//2`，`right = n_fft - win_length - left`。逐帧乘窗后做 `n_fft` 点 DFT，`onesided=True` 只保留 `n_fft//2+1` 个频点。输出布局是 `[freq][time]`，复数结果拆成 `{"real": ..., "imag": ...}`。

```text
x = pad(waveform, pad, "constant")
if center: x = reflect_or_constant_pad(x, n_fft // 2, pad_mode)
win = zero_pad_centered(window, n_fft, win_length)
seg[t] = [x[t * hop_length + j] * win[j] for j in range(n_fft)]
z[k][t] = sum_j seg[t][j] * exp(-2*pi*i*k*j/n_fft)
if frame_norm:  z /= sqrt(n_fft)
if window_norm: z /= sqrt(sum(window^2))
power is None -> (real, imag) ; power == 1.0 -> abs(z) ; else -> abs(z)^power
```

### 归一化分支

`_get_spec_norms` 把 `True` 与 `"window"` 映射成 `window_norm`（输出除以 `sqrt(sum(window^2))`，用**原始**窗口而非补零后的窗口，两者平方和相同），把 `"frame_length"` 映射成 `frame_length_norm`（传给 `torch.stft` 的 `normalized`，等价于除以 `sqrt(n_fft)`），`False` 两者都为假。非法字符串抛 `ValueError`，其它类型抛 `TypeError`。`return_complex` 非 `None` 只发弃用警告，不改变结果，因此返回值里的 `warnings` 长度应为 1。

### power 分支

`power is None` 返回复数；`power == 1.0` 返回 `abs(z)`；其它数值返回 `abs(z)**power`。`waveform` 支持 1D 与 2D（`(channel, time)`），2D 时输出形状为 `[channel, bins, frames]`。返回值还包含 `bins`、`frames`、`frame_norm`、`window_norm`、`normalized`、`window_padded`、`pad`、`center`、`pad_mode`、`onesided`、`power`、`warnings`。

### 域约束

`waveform` 只能是 1D 或 2D 且只含数字；`window` 必须是非空 1D 且长度等于 `win_length`；`n_fft`、`hop_length` 为正整数，`0 < win_length <= n_fft`；`pad` 为非负整数；`center`、`onesided` 为 `bool`；`pad_mode` 只能是 `reflect` / `constant`；`power` 是数字或 `None`。reflect 需要 `n_fft//2 < len(x)`，帧数必须大于 0，窗口能量必须为正。

## 8. 官方 quirk 清单

1. `amplitude_to_DB` 的 cutoff 是 per-batch，但 2D/3D 输入只有一个 batch。
2. `compute_deltas` 用互相关而非卷积，偶数 `win_length` 的 `n` 由 `(win_length-1)//2` 决定。
3. `sliding_window_cmn` 的切片上界笔误因首帧起点恒为 0 而不可观测。
4. 2D 单帧 CMN 输入会被 `squeeze(0)` 降维。
5. CMN 的 docstring 与源码对最后两维的描述互相矛盾。
6. `spectrogram` 的 `return_complex` 参数已废弃，只发警告。
7. `window_norm` 用原始窗口平方和；`frame_length_norm` 走 `stft(normalized=True)`。

## 9. 常见错误清单

- 把 4D 的 `top_db` 当成全局单 cutoff，或把 `packed_channels` 漏在 `chunk` 之外。
- 用 `win_length // 2` 代替 `(win_length - 1) // 2`，偶数窗口会错。
- 把 deltas 的 `conv1d` 写成卷积（核翻转），正负号会反。
- CMN 忘记 `min_cmn_window` 分支，或把 `window_end` 直接当窗口长度。
- reflect 左段从 `x[p-1]` 开始（正确是 `x[p]`），首帧 DFT 会错。
- 把 `normalized=True` 当成 `frame_length_norm`，或给补零后的窗口算平方和。
- 复数输出按 `[time][freq]` 排布，或把 `power=1.0` 与 `power=2.0` 混为一谈。
- 元数据字段写死常量，导致与数值口径不一致。

## 10. 自检

判题用例覆盖五条链路的公式、域约束、元数据与 quirk；参考解需通过 12 段官方探针、三层对拍（official / reference / 独立 oracle）与全量 mutation。任何未知 mode 必须抛 `ValueError`，消息包含 `unknown mode`。
