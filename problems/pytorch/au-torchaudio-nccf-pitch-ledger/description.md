# torchaudio NCCF 音高检测账：_compute_nccf 的四次方归一、_combine_max 的半窗偏好、_median_smoothing 的左侧 replicate pad 与频率换算的 float32 quirk

## 1. 背景

`torchaudio.functional.detect_pitch_frequency` 用归一化互相关（NCCF）逐帧估计基频。本题把整条链路拆成五个可独立判题的纯 Python 数值账：分帧互相关、阈值合并、半窗最大值、中值平滑、以及最终的频率换算。
本题与 `audio-yin-trough-parabolic-ledger`（838，librosa YIN：累积均值归一化差分 + 抛物线插值 + 谷值阈值）算法族完全不同；本题是 torchaudio 的 NCCF 互相关 + 半窗偏好 + 单侧中值平滑。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _compute_nccf          (lines ~1014-1058)
  _combine_max           (lines ~1061-1069)
  _find_max_per_frame    (lines ~1072-1099)
  _median_smoothing      (lines ~1102-1118)
  detect_pitch_frequency (lines ~1121-1163)
```

### 逐字执行环境

暂存目录 `.workbuddy/scratch/torchaudio858/official858.py` 用 `ast.get_source_segment` 从 855 已落盘的 2511 行官方文件中整段抽取上述五个顶层函数，`exec` 到共享命名空间；只注入 `math` / `typing` / `torch` / `Tensor` / `Tuple`，不 mock 任何函数体。驱动层统一 float64。

## 2. 口径约定

### 输入与输出

```text
solve(mode, **kw) -> dict
mode = nccf | combine_max | find_max | median_smooth | pitch
```

参考解暴露单入口 `def solve(mode, **kw)`；判题文件顶层是一组 `def check_*(solve)`，由 judge 注入 `solve`。判题用例不得 `from solution import ...`。
全部输入是嵌套 Python 列表，纯 Python 实现，不依赖 `torch` / `torchaudio` / `numpy`，不使用全局 RNG。任意前导维按 C 序展平为等长行，逐行计算后按 `lead + [tail]` 折回。

### 数值精度与非法输入

判题数值走 float64。★ 官方最后一行 `freq = sample_rate / (EPSILON + indices.to(torch.float))` 会把整数索引降级成 float32，`1e-9` 的 epsilon 被完全吞掉，索引 4 恰得 200.0；判题链路按同一公式在 float64 下重算，得 `199.99999995`，`freq_official` 单独暴露官方行为。
所有非法输入抛带具体关键字的 `ValueError`：`waveform must be a non-empty numeric array` / `waveform must be an array` / `sample_rate must be an integer` / `sample_rate must be at least 1` / `frame_time must be a positive number` / `freq_low must be an integer` / `freq_low must be at least 1` / `freq_high must be at least 1` / `win_length must be at least 3` / `nccf must have at least 2 dimensions` / `nccf last dimension is too short for freq_high` / `a and b must have the same length` / `thresh must be a positive number` / `indices length is too short for win_length` / `unknown mode`。校验顺序也属于契约。

## 3. mode 1：nccf

### 分帧与右侧补零

```text
lags       = ceil(sample_rate / freq_low)
frame_size = ceil(sample_rate * frame_time)
num_frames = ceil(waveform_length / frame_size)
pad        = lags + num_frames * frame_size - waveform_length
padded     = waveform + [0.0] * pad
s1 = padded[i * frame_size : i * frame_size + frame_size]
s2 = padded[lag + i * frame_size : lag + i * frame_size + frame_size]
```

`lag` 从 1 走到 `lags`（含），共 `lags` 列；`i` 从 0 走到 `num_frames - 1`，共 `num_frames` 行；输出形状 `lead + [num_frames, lags]`。`pad` 只补在右侧，尾帧会看到补零。

### 四次方归一化（与 docstring 相反）

```text
EPSILON = 1e-9
dot = sum(s1[k] * s2[k] for k in range(frame_size))
nr1 = sqrt(sum(a * a for a in s1))
nr2 = sqrt(sum(b * b for b in s2))
phi = dot / (EPSILON + nr1)**2 / (EPSILON + nr2)**2
```

★ 官方 docstring 写的是 `dot / sqrt(E1 * E2)`，源码实际是**两次连除两个平方范数**（四次方归一），因此数值远小于 1；`1e-9` epsilon 在 `0.2499999995 != 0.25` 上可见。★ 该归一化**不是尺度不变的**：波形整体乘 2 会让每个值除以 4（另有 epsilon 漂移），但 `argmax` 不变，故最终频率不变。

## 4. mode 2：combine_max

`_combine_max(a, b, thresh=0.99)` 逐元素做 `mask = a_values > thresh * b_values`，命中时**同时**取 `a` 的值与索引，否则取 `b`。★ 是严格大于：`a == 0.99 * b` 不命中。`thresh` 是本 mode 的显式入参，默认 `0.99`，必须为正数；四个数组等长且均为非空数值数组。

## 5. mode 3：find_max

`lag_min = ceil(sample_rate / freq_high)`；`best = max(nccf[..., lag_min:])`；`half_size = lags // 2`；`half = max(nccf[..., lag_min:half_size])`。官方以 `_combine_max(half, best)`（注意 `a` 是前半窗、**阈值硬编码 0.99**）偏好较小 lag，随后 `indices += lag_min` 再 `indices += 1` 两次校准。
`argmax` 取首次最大值以对齐 `torch.max`；`half_size <= lag_min` 时官方 `torch.max` 会在空张量上报错，故参考解提前抛 `nccf last dimension is too short for freq_high`。本 mode 要求 `nccf` 至少二维。

## 6. mode 4：median_smooth

`pad_length = (win_length - 1) // 2`，只做**左侧** replicate pad（右侧不补），再 `unfold(-1, win_length, 1)` 取滑窗，`torch.median` 在偶数窗取**下中位** `sorted[(n-1)//2]`。
★ 输出帧数 `n + pad_length - win_length + 1 = n - ceil((win_length - 1) / 2)`，比输入短；`win_length = 1 / 2` 时 `pad_length = 0`，官方 `torch.cat([])` 直接报错，因此参考解把域约束收窄为 `win_length >= 3`。

## 7. mode 5：pitch

先把批次 pack 成 `[-1, L]` 行，依次跑 nccf → find_max → median_smooth，最后 `freq = sample_rate / (1e-9 + index)`，再按输入前导维 unpack。中间账 `nccf` / `indices` / `smoothed` 一并返回，且按输入 rank 展开成非 packed 形状。
默认参数：`frame_time = 0.01`、`win_length = 30`、`freq_low = 85`、`freq_high = 3400`。`meta` 含 `lags` / `frame_size` / `num_frames` / `pad` / `padded_length`。

## 8. 返回字段一览

```text
nccf          : nccf shape waveform_length lags frame_size num_frames pad padded_length
combine_max   : values indices mask thresh
find_max      : indices lag_min half_size best_values best_indices half_values half_indices shape
median_smooth : smoothed padded roll pad_length shape
pitch         : freq shape nccf indices smoothed meta
```

## 9. 自测与判题

12 段官方探针 57/57；三层对拍（逐字官方 / 参考解 / 独立定义式 oracle）471 项全绿；顶层 `check_*(solve)` 共 21 个，参考解 21/21 ACCEPTED；变异测试原始 202 项，剔除 11 个真等价后 191/191 全 CAUGHT。
所有期望值均由官方逐字环境实测钉定，不使用容差掩盖机制差异；`scripts/check_bilingual.py` 负责全库中英一致性。
