# torchaudio biquad 系数族账：9 个 RBJ/SoX 架式滤波器、`bass` 预归一化、`band` 的 exp/sqrt/noise 分支、de-emphasis 与 RIAA 固定参数表、系数交给 `lfilter` 的 a-b 顺序

## 1. 背景

`torchaudio.functional.filtering` 里九个 `*_biquad` 函数各自把模拟原型经双线性变换成二阶 IIR 系数，最后统一调用 `biquad(waveform, b0, b1, b2, a0, a1, a2)`。该函数把 `torch.cat([a0,a1,a2])` 作 `a_coeffs`、`torch.cat([b0,b1,b2])` 作 `b_coeffs` 交给 `lfilter`。本题**只判系数推导**，用纯 Python 复刻成八个可独立判题的 mode（`coeffs` / `call` / `db2linear` / `presets` / `errors` / `census` / `grid` / `biquad_order`），把全部数值用官方常量快照钉死。
全库查重：九个系数函数名与 `_dB2Linear` 在 `problems` 的全部 `meta.json` / `description*.md` / `notes*.md` **零命中**。**861 `au-torchaudio-loudness-gate-ledger` 已完整覆盖** `biquad` / `lfilter` / `_lfilter` / `_lfilter_core_generic_loop` 与两级 K-weighting IIR 递推，判的是 ITU-R BS.1770-4 响度门控。故 869 **不重判 IIR 递推**：把 `lfilter` 换成显式入参 spy，只记录 `biquad` 交出的六个系数与 `clamp` / `batching` / `dtype`，手法同 868 的 `spectrogram` spy。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/filtering.py  (1702 lines)
  _dB2Linear(x)                       lines 12-13
  allpass_biquad(...)                 lines 70-105
  band_biquad(...)                    lines 108-158
  bandpass_biquad(...)                lines 161-204
  bandreject_biquad(...)              lines 207-241
  bass_biquad(...)                    lines 244-292
  biquad(...)                         lines 295-333
  deemph_biquad(...)                  lines 417-462
  equalizer_biquad(...)               lines 630-669
  lowpass_biquad(...)                 lines 1102-1132
  riaa_biquad(...)                    lines 1299-1360

  b0 = ...; b1 = ...; b2 = ...; a0 = ...; a1 = ...; a2 = ...
  return biquad(waveform, b0, b1, b2, a0, a1, a2)

  output_waveform = lfilter(waveform,
      torch.cat([a0, a1, a2]),   # a_coeffs FIRST
      torch.cat([b0, b1, b2]))  # b_coeffs SECOND
```

`official869.py` 用 `ast.get_source_segment` 抽出上述 11 个顶层函数并 `exec` 进同一共享 NS，**官方函数体一字未改**；注入对象只有真实的 `torch` / `math` / `warnings` / `typing` / `F`、`_IS_TORCHAUDIO_EXT_AVAILABLE=False` 与一个 `lfilter` spy。全局默认 dtype 固定为 `float64`。实测 `math.sin/cos` 与 `torch.sin/cos` 在本题输入域内差不超过 1.1e-16、`exp/sqrt` 完全一致，故纯 Python 参考解用 `math` 即可逐位对拍。

## 2. 口径约定

### 八个 mode

`solve(mode, **kw)` 分发八路：`coeffs` 返回 `{kind, a, b, a_norm, b_norm, a0_raw, waveform_shape, output_shape, params}` 完整账本，`call` 返回 spy 记录的 `waveform_shape` / `a_coeffs` / `b_coeffs` / `clamp=True` / `batching=True` / `dtype='torch.float64'`，`db2linear` 返回 `exp(x*ln10/20)`，`presets` 返回 2 个 deemph + 4 个 riaa 行，`errors` 返回 7 类边界，`census` 返回 14 条源码事实，`grid` 返回 9 个 kind × 各自合法采样率的所有行，`biquad_order` 用显式 `b0..a2` 直连 `biquad` 验系数顺序。

### 非有限值与输入域

torch 张量除法、开方、三角函数在边界给 `nan` / `±inf` 而**不抛异常**（Python `math` 会抛）。参考解必须显式模拟这套 IEEE 语义。为保证 fixtures 是合法 Python 字面量，所有非有限结果统一归一是字符串 `"nan"` / `"inf"` / `"-inf"`；`Q=0` 给 `a=[inf, ..., -inf]`，`sample_rate=0` 给全 `nan`，负采样率会翻转 inf 的符号，`band` 的 `Q=0` 给 `b0="nan"`。
`deemph_biquad` 只接受 44100 / 48000，`riaa_biquad` 只接受 44100 / 48000 / 88200 / 96000，其余采样率抛 `ValueError`（消息逐字对齐官方）；`grid` mode 对这两个 kind 只遍历合法率，其余七个 kind 遍历 8000 / 16000 / 44100。ragged 嵌套 `waveform` 复刻 `torch.tensor` 的维度消息 `expected sequence of length N at dim 1 (got M)`。

## 3. 九个 kind 的公式与 quirk

```text
w0    = 2*pi*freq / sample_rate
alpha = sin(w0) / 2 / Q          (RBJ cookbook)
A     = exp(gain / 40 * ln(10)) = _dB2Linear(gain/2)

lowpass     b = [(1-cos w0)/2, 1-cos w0, (1-cos w0)/2]
            a = [1+alpha, -2cos w0, 1-alpha]
allpass     b = [1-alpha, -2cos w0, 1+alpha]  == reverse(a)
bandreject  b = [1, -2cos w0, 1]
bandpass    temp = sin(w0)/2 if const_skirt_gain else alpha
            b = [temp, 0, -temp]  (skirt ratio == Q)
equalizer   b0/b2 = 1 +/- alpha*A ; a0/a2 = 1 +/- alpha/A
bass        SoX temp1/temp2/temp3 ; ALL SIX coefficients
            divided by a0 BEFORE calling biquad  (a0_raw == 1.0)
band        bw_Hz = freq/Q ; a2 = exp(-2*pi*bw_Hz/sr)
            a1 = -4*a2/(1+a2)*cos(w0) ; b1 = b2 = 0
            b0 = sqrt(1 - a1^2/(4*a2))*(1-a2)
            noise=True -> b0 = sqrt(((1+a2)^2-a1^2)*(1-a2)/(1+a2))
deemph      fixed (central_freq, width_slope, gain) per rate
            alpha uses sqrt((A+1/A)*(1/width-1)+2)
riaa        zeros/poles polynomials per rate, then g-scaled so
            |H(exp(j*2*pi*1000/sr))| == 1
```

① **`bass` 是唯一预先归一化的 kind**：先算六个未归一化系数，再全部 `/a0` 交给 `biquad`，所以 `a0_raw` 恒为 `1.0`、`a_norm == a`；其余八个 kind 传的是未归一化系数。② **`equalizer` 的 `alpha*A` 与 `alpha/A` 不对称**，故 `b != a`（`gain=0` 时 `A=1` 才恒等）；`bass` 在 `gain=0` 时同样退化为恒等滤波器。
③ **`band` 走的是另一套 SoX 架式**：`a0=1`、`a2=exp(-2π·bw_Hz/sr)`、`a1=-4a2/(1+a2)cos(w0)`、`b1=b2=0`，`noise=True` 时另乘 `mult` 因子；判别式 `1 - a1²/(4a2)` 在输入域内恒 ≥ 0。④ **`deemph` 与 `riaa` 用纯 Python `math`**（其余用 torch 张量），且各自的固定参数表只在指定采样率有效；`riaa` 另用 1 kHz 频响的实部/虚部算 `g` 做 0 dB 归一。
⑤ **`biquad` 把 a 放在前、b 放在后**：`lfilter(waveform, a_coeffs, b_coeffs, ...)` 收到 `a_coeffs=[a0,a1,a2]`、`b_coeffs=[b0,b1,b2]`。⑥ `Q<0` 或 `sample_rate<0` 会翻转 `alpha` 的符号，使 `lowpass` 的 `a0` 与 `a2` 互换；`cutoff_freq` 超过 Nyquist 也不报错，只是 `cos(w0)` 继续按公式取值。

## 4. 判题账本

`fixtures869.py` 从逐字官方环境导出 132 个常量，`cases869_checks.py` 提供 53 个 `check_*(solve)`：九个 kind 的默认系数与三档参数网格、`bass` / `equalizer` 各四档增益、`band` 的 noise 分支与负判别式 nan、`bandpass` 的 skirt 比例、`deemph` / `riaa` 全部合法预设与非法率消息、六个采样率的 `grid`、七类 `errors`、`call` 的 a-b 顺序与非有限值清洗、ragged 波形消息、混合符号 inf、`census` 的 14 条源码事实与「每次返回全新 dict」、`unknown mode` 的完整消息。
三层独立 oracle（`verify869.py`）不共享参考解算法：三角函数改由 `cmath.exp(1j*x)` 取实部/虚部、`A` 改由 `10**(gain/40)`、`exp` 改由 `cmath.exp`、`sqrt` 改由 `x**0.5`、π 改由 `4*atan(1)` 重建；RIAA 零点/极点多项式用 `Fraction` 精确有理数重算；再用 z 域传递函数代数性质（全通幅值恒 1、DC 单位增益、陷波、峰值增益、1 kHz 归一）交叉钉死。

## 5. 实现要点

参考解只用 `math`。`_div` / `_sin` / `_cos` / `_sqrt` / `_exp` 五件套模拟 torch 的 IEEE 语义，避免 Python 在除零与负数开方时抛异常；`_clean` 把非有限值转成稳定字符串；`_shape_of` 复刻 `torch.tensor` 的 ragged 检测；`_coeffs` 同时给出原始 a/b 与按 `a0` 归一化的 a_norm/b_norm。九种 kind 的公式各自独立实现，不共享中间量，便于逐项判题。
输入域内的浮点比较用 `rtol=1e-8`；`census` 与错误消息按字符串精确比较。除 `unknown mode` 抛 `ValueError` 外，所有异常在 `errors` mode 中收敛成 `{type, msg}` 字典，消息与官方逐字一致。
