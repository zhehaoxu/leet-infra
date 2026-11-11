# torchaudio 变速/变调编排账：rate 指数与 int 截断、hop/win/window 默认解算、按频点数取长度的 phase_advance linspace、len_stretch 的 banker's rounding、_fix_waveform_shape 的裁剪/补零与 view、speed 的 gcd 约简与 ceil+cast

## 1. 背景

`torchaudio.functional` 的 `pitch_shift`、`_stretch_waveform`、`_fix_waveform_shape` 与 `speed` 四个函数构成一层纯编排逻辑：它们不做任何数值 DSP，只负责解算参数默认值、计算速率与长度、约简采样率比、以及把结果形状修回原样。真正的重计算全部委托给 `torch.stft` / `torch.istft` / `phase_vocoder` / `resample`。本题把这层编排账用纯 Python 复刻并拆成十个可独立判题的 mode。
全库 `pitch_shift` / `_stretch_waveform` / `_fix_waveform_shape` / `speed` 在已立项题目的 meta、description、notes 中零命中（`pitch_shift` 仅在 860 的 notes 作为候选被提及）。857 覆盖 `phase_vocoder` 本体、859 覆盖 sinc 重采样 kernel 本体、860 覆盖 `torch.istft` / Griffin-Lim、862 与 863 覆盖 MVDR 与 FID，函数集合不重叠。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  speed                    (line 2385, 39 lines)
  pitch_shift              (line 1596, 46 lines)
  _stretch_waveform        (line 1644, 50 lines)
  _fix_waveform_shape      (line 1696, 24 lines)

  pitch_shift:
    waveform_stretch = _stretch_waveform(...)
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)
    return _fix_waveform_shape(waveform_shift, waveform.size())

  _stretch_waveform:
    hop = n_fft // 4 ; win = n_fft ; hann_window(win) if window is None
    waveform = waveform.reshape(-1, shape[-1])
    spec_f = torch.stft(..., center=True, pad_mode='reflect',
             normalized=False, onesided=True, return_complex=True)
    phase_advance = torch.linspace(0, pi * hop, spec_f.shape[-2])[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    return torch.istft(spec_stretch, ..., length=len_stretch)

  _fix_waveform_shape:
    trim to ori_len, else zero-pad, then view(shape[:-1] + [ori_len])

  speed:
    src = int(factor * orig_freq) ; tgt = int(orig_freq)
    gcd = math.gcd(src, tgt) ; src, tgt = src // gcd, tgt // gcd
    out_lengths = ceil(lengths * tgt / src).to(lengths.dtype)
    return resample(waveform, src, tgt), out_lengths
```

### 逐字执行环境与注入口径

`official864.py` 用 `ast.get_source_segment` 从落盘的 2511 行官方源码抽出四个函数定义，`exec` 进同一个共享命名空间，官方函数体一字未改。四个重 DSP 依赖做**记录型注入**：`torch` 与 `math` 是转发代理，只额外记录 `stft` / `istft` / `gcd` 的调用实参；`phase_vocoder` 与 `resample` 是记录型包装，返回调用方显式给定的张量。于是 rate 指数、`int()` 截断、gcd 约简、`round()` 半偶舍入、linspace 长度、默认值解算、pack/view、ceil+cast、trim/pad 全部真实执行，而中间参数从调用记录里取作权威 ledger。
这个剥除是刻意的：STFT/ISTFT 的 reflect padding、相位声码器的极坐标累加、sinc kernel 的窗化与多相卷积分别由 860 / 857 / 859 逐字覆盖；把它们作为显式入参后，本题判题域不含 FFT、不含随机性、不含音频 IO，纯 Python 参考解可以在毫秒级与官方逐位对拍。

## 2. 口径约定

### 输入

- 所有参数都是 `solve(mode, **kw)` 的关键字参数，均有默认值；`waveform`、`spec_f`、`stretched`、`shifted`、`fix_waveform`、`speed_waveform`、`speed_out`、`window`、`speed_lengths` 是显式入参的嵌套数值列表（int / float，bool 被拒绝）。
- 核心标量：`n_steps=4`、`bins_per_octave=12`、`n_fft=16`、`sample_rate=16000`、`speed_factor=1.5`、`speed_orig_freq=16000`；`win_length` / `hop_length` / `window` / `speed_lengths` 默认 `None`。
- `spec_f` 至少二维；官方取 `spec_f.shape[-2]` 作为频点数，与帧数（`shape[-1]`）无关。默认注入 `(n_fft//2+1) x 3` 的零谱。
- `fix_target_shape` 是目标形状列表；`speed_lengths_dtype` 取 `float64` / `float32` / `int64` / `int32`，决定 `ceil` 结果 `.to(dtype)` 后是 int 还是 float。

### 十个 mode

1. `rate` → `2.0 ** (-n_steps / bins_per_octave)`，来自 `pitch_shift` 与 `_stretch_waveform` 共用的速率。
2. `pitch_rates` → `[int(sample_rate / rate), sample_rate]`，注意第一项是**截断**而非四舍五入。
3. `stretch_plan` → `[hop_length, win_length, window_len, freq_bins]`，四个默认值解算结果。
4. `phase_advance` → `linspace(0, pi*hop_length, freq_bins)` 的完整数值列表。
5. `len_stretch` → `int(round(ori_len / rate))`，Python 半偶舍入。
6. `pack_shape` → `waveform.reshape(-1, ori_len)` 之后的 `[prod(lead), ori_len]`。
7. `fix_shape` → 独立的 `_fix_waveform_shape` 输出波形（可用 `fix_waveform` / `fix_target_shape` 单独配置）。
8. `pitch_shift` → 端到端输出，等于对注入的 `shifted` 波形做 `_fix_waveform_shape`。
9. `speed_rates` → `[int(factor*orig_freq), int(orig_freq), gcd, src//gcd, tgt//gcd]`。
10. `speed` → `[speed_out, out_lengths]`，其中 `out_lengths` 为 `None` 或 `ceil(lengths*tgt/src)` 列表。

## 3. 官方机制逐段

### 3.1 rate 与 int 截断

`rate = 2.0 ** (-float(n_steps) / bins_per_octave)`：`n_steps` 被显式 `float()`，所以传 `5.5` 这样的半音步数合法（签名写的是 int，但函数体不校验）。`bins_per_octave` 不必是 12，`36` 表示每八度 36 档。随后 `pitch_shift` 用 `int(sample_rate / rate)` 作为重采样的源采样率：`n_steps=4, sample_rate=16000` 时 `16000/0.7937005... = 20158.738...`，官方给 **20158**（`round` 会得 20159，`ceil` 会得 20159）。

### 3.2 默认解算与按频点数的 linspace

`_stretch_waveform` 只在参数为 `None` 时补默认：`hop_length = n_fft // 4`、`win_length = n_fft`、`window = torch.hann_window(win_length)`；显式给 `hop_length` / `win_length` / `window` 时不校验一致性，所以 `window` 长度可以任意。随后 `phase_advance = torch.linspace(0, pi * hop_length, spec_f.shape[-2])[..., None]`——**长度取的是频点数 `n_fft//2+1`，不是时间帧数**；端点由 `hop_length` 决定，第 i 项是 `pi*hop*i/(freq_bins-1)`。

### 3.3 banker's rounding

`len_stretch = int(round(ori_len / rate))` 用的是 Python 内建 `round`（ties-to-even），不是 `floor(x+0.5)`。取 `n_steps=-12` 使 `rate=2`：`ori_len=9` → `4.5 → 4`；`15` → `7.5 → 8`；`5` → `2.5 → 2`；`7` → `3.5 → 4`。`torch.istft` 收到的 `length` 就是这个值。

### 3.4 trim / pad 与 view

`_fix_waveform_shape` 比较 `shift_len` 与 `ori_len`：过长则 `waveform_shift[..., :ori_len]` 裁剪，过短则 `F.pad(..., [0, ori_len - shift_len])` 在末尾补零，最后 `view(shape[:-1] + [ori_len])` 把批前导维填回。因为先统一到 `ori_len`，view 的目标形状恒等于原始 `waveform.size()`；三维前导 `[2,3]`、一维无前导都要正确重建。

### 3.5 speed 的 gcd 与 ceil+cast

`speed` 先 `src = int(factor*orig_freq)`、`tgt = int(orig_freq)`（**两次截断**），再用 `math.gcd` 约简，把约简后的两个频率交给 `resample`。`factor=1.5, orig_freq=16000` → `24000/16000` → gcd 8000 → `3/2`；`factor=0.7, orig_freq=44100` 因浮点误差得 `30869.999...`，截断成 **30869** 而不是 30870，gcd 因此变成 1。有效长度走 `torch.ceil(lengths * tgt / src).to(lengths.dtype)`：先逐元素 ceil，再 cast 回原 dtype；整型 lengths 得到整数列表，浮点 lengths 得到整数值的 float 列表，`lengths=None` 时返回 `None`。

## 4. 代码骨架

```python
def solve(mode, **kw):
    c = _cfg(kw)                      # apply all defaults
    L = _ledger(c)                    # one shared ledger for all modes
    if mode == "rate":
        return 2.0 ** (-n_steps / bins_per_octave)
    if mode == "pitch_rates":       # int() truncation, not round()
        return [int(sample_rate / rate), int(sample_rate)]
    if mode == "len_stretch":       # Python round is ties-to-even
        return int(round(ori_len / rate))
    if mode == "speed_rates":
        g = math.gcd(int(factor * orig_freq), int(orig_freq))
        return [src_raw, tgt_raw, g, src_raw // g, tgt_raw // g]
    ...                                # seven more modes
```

## 5. 关键 quirk 与易错点

1. `int(sample_rate / rate)` 与 `int(sample_rate * 2**(n/12))` 数学等价，但浮点上可能差一；本题钉死前者。
2. `phase_advance` 的 linspace 长度来自 `spec_f.shape[-2]`（频点），注入谱的行数直接决定输出长度，与帧数无关。
3. `round` 是半偶舍入，`4.5 -> 4` 而 `7.5 -> 8`；用 `floor(x+0.5)` 会在半数点错一半。
4. `speed` 传给 `resample` 的是 **gcd 约简后** 的频率，不是 `int(factor*orig_freq)` 的原始值。
5. `_fix_waveform_shape` 只在最后一维裁剪/补零；前导批发维通过 `view` 重建，补零恒为尾部零。
6. `n_steps` 是 `float()` 化的，分数步数合法；`bins_per_octave` 为 0 会抛 `ZeroDivisionError`。
7. `stretched`（istft 输出）在编排层只被校验、不被消费：重采样返回的是另一个独立注入的 `shifted`。

## 6. 验证流水线

官方逐字环境冒烟通过 → 12 段探针 `427/427 ALL GREEN` → 三层对拍 `13326 checks / 52 configs ALL GREEN`（第三层独立 oracle 用 `exp+log` 代替 `pow`、位移代替整除、`Fraction` 步长、`Decimal ROUND_HALF_EVEN`、手写欧几里得、`Decimal ROUND_CEILING`、索引驱动的分块 reshape）→ 判题 driver `152/152 ALL ACCEPTED` → 变异测试 160 项剔除 13 个文档化真等价后 `147/147 全 CAUGHT / 0 SURVIVED / 0 ERROR`。

## 7. 剥除口径与注册

判题域不含 FFT、复数谱数值、窗函数数值、sinc 卷积、音频文件与真实模型；所有波形/谱/窗全部显式入参，无 torch / numpy / scipy / 全局 RNG。四份双语文档严格行对齐，`register864.py` 把 `audio` 从 15 增长到 16、题数从 841 到 842、`max_order=864`，中英合集描述各追加 864 口径段。按约定改完先不 `git commit`，等用户确认。
