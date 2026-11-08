# Kaldi 兼容前端口径账：mel 的 1127/700 口径、VTLN 三段线性四次赋值覆盖、get_mel_banks 的 warp 双分支与 (num_bins,1) 形状 quirk、_get_strided 的 snip_edges 反射填充、前处理链顺序与 raw_energy，以及 fbank/mfcc 的开关口径与 htk_compat 尾部编排

## 1. 背景

torchaudio 的 `compliance/kaldi.py` 是 Kaldi fbank / mfcc 特征的逐位兼容实现。本题把整条前端链路拆成五个可独立判题的数值账，全部用纯 Python 浮点实现，dither 噪声一律显式入参，不依赖任何随机源。与 order 340-343 的基础组件题不同，本题钉死的是兼容层的参数口径与官方 quirk，而不是单个窗函数或滤波器组的定义。

### 官方源码位置

- `src/torchaudio/compliance/kaldi.py`：`_feature_window_function` / `_get_strided` / `_get_window` / `_get_log_energy` / `get_mel_banks` / `vtln_warp_freq` / `vtln_warp_mel_freq` / `spectrogram` / `fbank` / `mfcc`
- `src/torchaudio/functional/functional.py::create_dct`：mfcc 的 DCT 矩阵来源

判题环境没有 `torchaudio`，官方逐字执行环境靠注入 fake `torchaudio.functional.create_dct` 建立，探针跑的是真正的官方实现而不是重写版。

## 2. 口径约定

### 输入与输出

波形是一维列表 `[n]`，分帧结果是 `[m][padded_window_size]`，滤波器组是 `[num_bins][num_fft_bins]`，特征是 `[m][d]`。每个 mode 返回字典，字段名与官方变量对齐。所有非法输入统一抛 `ValueError`，消息点名被违反的约束。

### 数值精度与 float32 quirk

全程 Python float（float64）。`EPS` 恒为 `torch.finfo(torch.float).eps = 1.1920928955078125e-07`：官方 `EPSILON` 是 float32 标量，`_get_epsilon` 只做 `.to(dtype)`，数值不随 dtype 变化。

官方 `get_mel_banks` 内部 `torch.arange(int64) * float` 会降为 float32，即使输入 float64，`center_freqs` 在几千 Hz 处仍有约 `1e-3` 的量化误差，经 DCT 累积后把 mfcc `features` 推到约 `2.5e-5` 的绝对误差。因此 official-vs-reference 默认用 `tol=1e-6, rel=1e-5`，mfcc `features` 单独放宽到 `tol=5e-5, rel=0.0`；oracle-vs-reference 始终用 `1e-7`。

## 3. mode 1: vtln_warp

### 三段线性与四次赋值覆盖

`vtln_warp_freq` 先算两个拐点与左右斜率，再对整段频率做四次相互覆盖的赋值：

```text
l = vtln_low * max(1.0, vtln_warp)
h = vtln_high * min(1.0, vtln_warp)
scale = 1.0 / vtln_warp
Fl = scale * l
Fh = scale * h
scale_left = (Fl - low_freq) / (l - low_freq)
scale_right = (high_freq - Fh) / (high_freq - h)
```

官方源码注释明确写着 `order of operations matter here (since there is overlapping frequency regions)`。四次赋值依次是 ① `after_h`（`freq >= h`）写 `high_freq + scale_right*(freq-high_freq)`；② `before_h`（`freq < h`）写 `scale*freq`；③ `before_l`（`freq < l`）写 `low_freq + scale_left*(freq-low_freq)`，覆盖 ② 的低频重叠段；④ `outside_low_high_freq`（`freq < low_freq` 或 `freq > high_freq`）最后把区间外频率原样写回。最终 `F(low_freq)=low_freq`、`F(high_freq)=high_freq`，中段 `l <= f <= h` 是 `f/vtln_warp`。`vtln_warp_mel_freq` 是 `mel_scale(vtln_warp_freq(inverse_mel_scale(mel)))`，warp 不为 1 时 left / center / right mel 三列全部弯折。

### 域约束

`vtln_low <= low_freq` 抛 `vtln_low must be higher than low_freq`；`vtln_high >= high_freq` 抛 `vtln_high must be lower than high_freq`；`vtln_warp <= 0` 抛 `vtln_warp must be positive`；拐点越界抛 `inflection points out of range`。

## 4. mode 2: mel_banks

### mel 尺度与 delta

mel 是 `1127*ln(1+f/700)`、逆是 `700*(exp(m/1127)-1)`，不是 HTK 的 `2595*log10(1+f/700)`。

```text
mel_low_freq = 1127.0 * ln(1 + low_freq / 700.0)
mel_high_freq = 1127.0 * ln(1 + high_freq / 700.0)
mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
```

`mel_freq_delta` 的分母是 `num_bins + 1`。`left_mel` / `center_mel` / `right_mel` 分别是 `mel_low_freq + bi*delta` / `+(bi+1)*delta` / `+(bi+2)*delta`。`num_fft_bins = window_length_padded / 2`，`fft_bin_width = sample_freq / window_length_padded`，mel 网格是 `mel(fft_bin_width*k)`。`high_freq <= 0` 时加 nyquist，`vtln_high < 0` 时加 nyquist。

### warp 双分支

```text
up = (mel[k] - lm) / (cm - lm)
down = (rm - mel[k]) / (rm - cm)
warp == 1.0 : max(0.0, min(up, down))
warp != 1.0 : lm < mel[k] <= cm -> up ; cm < mel[k] < rm -> down ; else 0.0
```

`vtln_warp == 1.0` 走 `torch.max(zeros, torch.min(up_slope, down_slope))`；`vtln_warp != 1.0` 走 mask 分支，边界用严格 / 非严格不等号区分。两条分支数值不同，不能互相顶替。

### center_freqs 形状 quirk

官方 docstring 声称 `center_freqs` 形状是 `(num_bins,)`，实际返回是 `(num_bins, 1)`；本题保留双层列表。`padded_window_size` 必须为偶数且不小于 `window_size`，`num_bins` 必须大于 3。

## 5. mode 3: framing

### 分帧公式

```text
snip_edges=True  : m = 1 + (num_samples - window_size) // window_shift
snip_edges=False : m = (num_samples + window_shift // 2) // window_shift
pad = window_size // 2 - window_shift // 2
```

`snip_edges=True` 且波形短于窗长时返回 0 帧（`(0,0)` 语义）。`snip_edges=False` 用 `waveform[::-1]` 做反射填充：`pad > 0` 时拼接 `reversed[-pad:] + waveform + reversed`，`pad <= 0` 时拼接 `waveform[-pad:] + reversed`。反射填充不是零填充，也不是 `replicate`。

### 前处理链顺序

```text
dither -> remove_dc_offset -> raw_energy -> preemphasis -> window
       -> right zero pad -> post_energy (if not raw_energy)
```

- `dither`：`noise[i][j] * dither`，噪声是显式入参。
- `remove_dc_offset`：每帧减均值（除以 `window_size`）。
- `raw_energy=True` 在 preemphasis 与加窗**之前**算能量，`False` 在加窗与补零**之后**算。
- `preemphasis`：官方 `F.pad(..., (1,0), mode='replicate')[:, :-1]`，首样本取 `x[0]`，故 `j=0` 是 `(1-coeff)*x0`。
- `_get_log_energy`：`log(max(sum(x^2), EPS))`，`energy_floor` 在 **log 域**取 `max(le, log(energy_floor))`；`energy_floor=0` 直接返回 `le`。
- 窗函数五分支：`hamming` / `hanning` / `povey` / `rectangular` / `blackman`；`povey = hann^0.85`；blackman 是官方手写公式，默认 `blackman_coeff=0.42` 时恰等于 `torch.blackman_window`，`0.5` 时不等；非法窗抛 `Invalid window type`。

### 默认值

```text
snip_edges=True, window_type=povey, blackman_coeff=0.42, dither=0.0
remove_dc_offset=True, preemphasis_coefficient=0.97
raw_energy=True, energy_floor=1.0
```

全零波形下默认 `energy_floor=1.0` 必须把 log-energy clamp 到 `log(1)=0`；`povey` 窗不依赖 `blackman_coeff`，验证默认系数必须显式走 blackman 分支。

## 6. mode 4: fbank

### 频谱与 mel 能量

`_get_window` 的分帧结果逐帧做 `torch.fft.rfft(x).abs()` 取 `k=0..N/2`，`use_power` 时平方得 `spectrum`；`bins` 右侧补一列零成 `(num_bins, num_fft_bins+1)` 后与 spectrum 内积得 `mel_energies`；`use_log_fbank` 时取 `log(max(v, EPS))`。`use_energy` 时拼接 `signal_log_energy`，`subtract_mean` 时按列减均值做 CMS。

### 开关口径

```text
use_power      : spectrum = |X|^2   else |X|
use_log_fbank  : log(max(E, EPS))   else E
use_energy     : htk_compat -> [E..., sig] else [sig, E...]
subtract_mean  : per-column mean subtraction over frames
```

官方 `spectrogram` 的第 0 列会被 `signal_log_energy` 覆盖，不是 `log(|X0|^2)`；本题把能量单独作为 `signal_log_energy` 字段返回。

## 7. mode 5: mfcc

### DCT 矩阵与 lifter

```text
dct = create_dct(num_mel_bins, num_mel_bins, "ortho")
dct[:, 0] = sqrt(1 / num_mel_bins)
dct = dct[:, :num_ceps]
lifter[i] = 1 + 0.5 * L * sin(pi * i / L)
```

mfcc 强制 `use_power=True` / `use_log_fbank=True` / `subtract_mean=False` 调 fbank。`_get_dct_matrix` 用 `create_dct(..., 'ortho')` 转置后把首列强制覆盖成 `sqrt(1/n)`（覆盖掉 ortho 的 `1/sqrt(2)` 因子）。lifter 的 C0 恒为 `1.0`，`cepstral_lifter=0.0` 时官方跳过 lifter（本题同步跳过，避免除零）。`num_ceps > num_mel_bins` 抛 `num_ceps cannot be larger than num_mel_bins`。

### htk_compat 尾部编排

```text
use_energy: sig = row[num_mel_bins if htk_compat else 0]
            mel_offset = 0 if htk_compat else 1
            C0 = sig (overwrite)
htk_compat: energy = row[0] (or row[0] * sqrt(2.0) if not use_energy)
            out = row[1:] + [energy]
```

`use_energy` 时先把能量列从 fbank 特征里摘出（`htk_compat` 在末尾、否则在首列），DCT 后把 C0 覆写成该能量；`htk_compat` 再把 C0/energy 移到末尾。最后 `subtract_mean` 才作用到 mfcc 输出上。

## 8. 官方 quirk 清单

1. `center_freqs` 实际是 `(num_bins, 1)`，docstring 写 `(num_bins,)`。
2. VTLN 四次赋值靠重叠区间覆盖，顺序不可换；区间外频率原样保留。
3. `get_mel_banks` 内部 int64 乘 float 降为 float32，`center_freqs` 有约 `1e-3` 量化误差。
4. `spectrogram` 第 0 列被 `signal_log_energy` 覆盖，不是 `log(|X0|^2)`。
5. 默认 `blackman_coeff=0.42` 时官方手写公式恰等于 `torch.blackman_window`。
6. `energy_floor` 在 log 域取 max，不是能量域 clamp。
7. `snip_edges=False` 用反射填充而非零填充。
8. mel 是 `1127/700` 自然对数口径，不是 HTK 的 `2595/log10`。
9. `cepstral_lifter=0.0` 时跳过 lifter，C0 恒 `1.0`。
10. warp 不为 1 时 left / center / right mel 全部弯折，bins 与 center_freqs 都变化。

## 9. 常见错误清单

1. 把 mel 写成 HTK 的 `2595*log10(1+f/700)`。
2. `mel_freq_delta` 除以 `num_bins` 而不是 `num_bins + 1`。
3. 把 `center_freqs` 展平成 `(num_bins,)`。
4. VTLN 四次赋值顺序调换或漏掉 `outside_low_high_freq` 的最后覆盖。
5. `l` 用 `min` / `h` 用 `max`，或 `scale` 写成 `vtln_warp`。
6. `warp != 1.0` 时仍用 `max/min` 分支，或漏掉 mask 边界的严格不等号。
7. `warp != 1.0` 时忘弯折 left / right mel，只弯 center。
8. `snip_edges=False` 用零填充代替反射填充。
9. `snip_edges=True` 波形过短没返回 0 帧。
10. `pad` 的 `//` 写成浮点除法。
11. 前处理链顺序错：把 `raw_energy` 算在加窗之后。
12. `preemphasis` 首样本取 `x[-1]` 或 `0`，而不是 replicate 的 `x[0]`。
13. `energy_floor` 在能量域 clamp 而不是 log 域。
14. `povey` 忘了 `**0.85`。
15. blackman 手写系数写错符号或角度。
16. `dither` 噪声漏乘 `dither` 或没用显式 `noise`。
17. `bins` 忘了右侧补零列。
18. `use_energy` 的能量列在 `htk_compat` 下放错首/尾。
19. `dct` 首列忘了强制 `sqrt(1/n)`。
20. `cepstral_lifter=0.0` 时仍算 lifter 导致除零。
21. `htk_compat` 没把 C0/energy 移到末尾。
22. 默认 `energy_floor=1.0` / `blackman_coeff=0.42` 被改掉。
23. 未知 mode 不抛 `unknown mode`。
24. 域约束只断言「抛异常」而不看消息关键字。

## 10. 自检

- 全零波形默认 `energy_floor=1.0` 时 `signal_log_energy[0] == log(1.0) == 0`。
- mel 与 inverse mel 在正频率上互逆。
- `warp=1` 与 `warp!=1` 的 `bins` / `center_freqs` 必须不同。
- `num_ceps <= num_mel_bins`，`lifter_coeffs[0] == 1.0`。
- 三层对拍、判题用例、mutation 全部通过后才落地。
