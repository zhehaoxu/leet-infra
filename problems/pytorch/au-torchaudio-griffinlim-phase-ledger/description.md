# torchaudio Griffin-Lim 相位重建账：norms / dtype 分流、reflect 填充的分帧 DFT、Hermitian 镜像与窗包络 OLA、pad 只在 length 给定时生效、momentum 的原地 tprev 与 +1e-16 归一化

## 1. 背景

`torchaudio.functional.griffinlim` 从幅度谱反复重建相位，`inverse_spectrogram` 把复数谱经 `torch.istft` 还原波形。本题把两者依赖的 `torch.stft` / `torch.istft` 也用纯 Python DFT + overlap-add 显式重写，拆成六个可独立判题的数值账：归一化分流、复数 dtype 映射、前向分帧 DFT、Hermitian 逆 DFT 与 OLA、inverse_spectrogram 编排、以及完整 Griffin-Lim 迭代。
本题与 855 / 857 / 858 / 859 同源于 `functional.py`，但函数集合互不重叠；`mask_along_axis` 已被 order 693 的 `asr-specaugment-mask` 覆盖故避让。全库 `griffinlim` / `inverse_spectrogram` 零命中。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  inverse_spectrogram  (lines ~148-205)
  _get_spec_norms      (lines ~208-223)
  _get_complex_dtype   (lines ~226-233)
  griffinlim           (lines ~236-353)
```

### 逐字执行环境

暂存文件 `.workbuddy/scratch/torchaudio860/official860.py` 用 `ast.get_source_segment` 从 855 已落盘的 2511 行官方文件中整段抽取上述四个顶层函数，`exec` 到共享命名空间；只注入 `math` / `typing` / `warnings` / `torch` / `Tensor` / `Optional` / `Union` / `_CPU`，不 mock 函数体。驱动层统一 float64 / complex128。为把 `rand_init=True` 的初相位显式入参化，仅在驱动层临时把 `torch.rand` 换成固定复数矩阵（`angles_override`），官方函数体本身不改。

## 2. 口径约定

### 输入与输出

```text
solve(mode, **kw) -> dict
mode = norms | complex_dtype | stft | istft | inverse | griffinlim
```

参考解暴露单入口 `def solve(mode, **kw)`；判题文件顶层是一组 `def check_*(solve)`，由 judge 注入 `solve`，不得 `from solution import ...`。全部输入是嵌套 Python 列表，纯 Python 实现，不依赖 `torch` / `torchaudio` / `numpy`，不使用全局 RNG。
★ 判题域只取二维 `(freq, frames)` 谱，批次维不纳入。复数矩阵用两份实矩阵 `real` / `imag` 表达；`angles_init` 用 `[re, im]` 对的矩阵显式给出，替代官方 `torch.rand(complex128)` 的随机初相位。`rand_init=False` 对应全 1 复数初相位，`rand_init=True` 对应把固定 seed 下 `torch.rand` 的逐值矩阵作为入参传入。
判题数值全程 float64。参考解用 `cmath.exp` 的复指数做 DFT，独立 oracle 用实余弦/正弦展开 Hermitian 逆 DFT，并以「逐输出样本 gather 各帧」代替 scatter；两者与官方误差约 1e-15。

### 非法输入

所有非法输入抛带具体关键字的 `ValueError`（`norms` 的类型错误抛 `TypeError`）：`must be a non-empty list of numbers` / `must contain only numbers` / `must be a non-empty 2-D list` / `must have at least one column` / `must be rectangular` / `must be an int` / `must be >=` / `must be a number` / `must be a bool` / `win_length must be <= n_fft` / `hop_length must be <= win_length` / `window length must equal win_length` / `freq dimension must be n_fft // 2 + 1` / `real and imag shapes must agree` / `Invalid normalized parameter` / `Input type not supported` / `Unexpected dtype` / `power must be > 0` / `momentum must be in range` / `reflect pad requires` / `waveform shorter than n_fft` / `window overlap add min` / `rebuilt frame count does not match specgram` / `unknown mode`。校验顺序与精确消息都属于契约。

## 3. mode 1：norms / mode 2：complex_dtype

`_get_spec_norms` 把 `normalized` 映射成 `(frame_length_norm, window_norm)` 两个布尔。字符串只接受 `"frame_length"` 与 `"window"`；布尔 `True` 等价于字符串 `"window"`；`False` 两者皆否；其它类型抛 `TypeError`。`_get_complex_dtype` 只认 `double`/`float64` → `complex128`、`float`/`float32` → `complex64`、`half`/`float16` → `complex32`，其它抛 `ValueError`。

```text
normalized=False        -> (False, False)
normalized=True         -> (False, True)
normalized='window'     -> (False, True)
normalized='frame_length' -> (True, False)
double / float64 -> torch.complex128
float  / float32 -> torch.complex64
half   / float16 -> torch.complex32
```

## 4. mode 3：stft（reflect 填充 + 分帧 DFT）

★ `center=True` 的 reflect 填充**不含边界样本**：左侧是 `x[1:p+1][::-1]`，右侧是 `x[-p-1:-1][::-1]`，其中 `p=n_fft//2`；因此 `[1..8]` 在 `n_fft=8` 下补成 `[5,4,3,2,1,2,3,4,5,6,7,8,7,6,5,4]`，首帧 DC 是 24 而不是 20。要求 `len(x) > p`。`win_length < n_fft` 时窗左补 `(n_fft-win_length)//2` 个零、右侧补剩余个数。

```text
p = n_fft // 2
padded = x[1:p+1][::-1] + x + x[-p-1:-1][::-1]
T = 1 + (len(padded) - n_fft) // hop_length
X[k,t] = sum_n padded[t*hop+n] * w[n] * exp(-2*pi*i*k*n/n_fft)
k = 0 .. n_fft//2 ;  normalized=True -> X / sqrt(n_fft)
left_zero = (n_fft - win_length) // 2
```

输出含 `shape=[n_fft//2+1, T]`、`real` / `imag` 两账本矩阵、`n_frames`、`padded_length`、`n_fft` / `hop_length` / `win_length` / `center` / `normalized`。

## 5. mode 4：istft（Hermitian 镜像 + OLA + 窗包络除法）

单帧 `col` 长度 `K=n_fft//2+1`，先按 `full = col + [conj(col[n_fft-k]) for k in K..n_fft-1]` 镜像成 `n_fft` 点 Hermitian 谱，再做带 `1/n_fft` 的逆 DFT 取实部；`torch.fft.irfft` 会丢弃 DC 与 Nyquist 的虚部，取实部的写法自动等价。每帧乘窗后按 `hop_length` 累加进缓冲，同时把 `w[n]^2` 累加进 envelope，最后逐样本除以 envelope。

```text
total = (T-1)*hop_length + n_fft
buf[t*hop+n] += irfft(col_t)[n] * w[n]
env[t*hop+n] += w[n] * w[n]
start = n_fft//2 if center else 0
end   = total - (n_fft//2 if center else 0)
size  = length if length is not None else end - start
out[j] = buf[start+j] / env[start+j]
normalized=True -> buf *= sqrt(n_fft)
i >= total -> out = 0.0 , env = 1.0  (zero padding)
min(env) < 1e-11 -> ValueError('window overlap add min')
```

★ 三点官方 quirk：① `center=True` 只是从 `n_fft//2` 起取，末尾裁掉 `n_fft//2`；若给了 `length` 就直接从 `start` 取 `length` 点，超出缓冲补零且 envelope 补 1。② `normalized=True` 是整体乘 `sqrt(n_fft)`（`stft` 则是除）。③ 对称 Hann 窗首尾为零，当 `length` 把输出拉到窗边缘之外时 envelope 出现 0，`torch.istft` 抛 `window overlap add min`；period  Hann 或不给 `length` 则安全。

## 6. mode 5：inverse（inverse_spectrogram 编排）

先取 `(frame_length_norm, window_norm)`；若 `window_norm` 为真，把整个复数谱乘以 `sqrt(sum(window^2))`（用的是**未补零**的 `win_length` 长窗）；然后把 `frame_length_norm` 作为 `istft` 的 `normalized` 传入；`length` 给定时把 `length + 2*pad` 传给 istft，再切片 `waveform[pad:-pad]`。★ `pad` 只在 `length` 给定时生效；`pad_mode` 在官方签名里存在但从未被使用。

```text
frame_length_norm, window_norm = _get_spec_norms(normalized)
scale = sqrt(sum(w^2)) if window_norm else 1.0
spec  = spec * scale
istft_length = length + 2*pad if length is not None else None
wave = torch.istft(spec, ..., normalized=frame_length_norm,
                   length=istft_length)
if length is not None and pad > 0: wave = wave[pad:-pad]
```

`normalized=True` / `'window'` 乘 `sqrt(sum(w^2))`，`'frame_length'` 走 `istft(normalized=True)` 乘 `sqrt(n_fft)`；矩形窗 `n_fft=8` 时两者数值恰好相同，但机制不同。输出含 `out` / `shape` / `target_length` / `frame_length_norm` / `window_norm` / `window_scale` / `istft_length` / `pad` / `envelope_min`。

## 7. mode 6：griffinlim（相位迭代）

```text
if not 0 <= momentum < 1: raise ValueError
effective = momentum / (1 + momentum)
mag    = specgram ** (1 / power)
angles = angles_init or all-ones complex
tprev  = scalar 0.0
repeat n_iter times:
  inverse = istft(mag * angles, length=length)
  rebuilt = stft(inverse)                 # center=True, reflect
  angles  = rebuilt
  if effective: angles = angles - tprev.mul_(effective)
  angles  = angles / (abs(angles) + 1e-16)
  tprev   = rebuilt                     # UNnormalised
return istft(mag * angles, length=length)
```

★ 五个高价值 quirk：① `momentum` 先被原地改写为 `m/(1+m)` 才使用。② `tprev.mul_(m)` 是原地操作，但 `angles.div(...)` 是**非原地**的，所以 `tprev = rebuilt` 保存的是未归一化的复数谱；参考解须保存归一化前的副本。③ 除数里的 `+1e-16` 让零幅谱不致 `0/0`（官方全零初相位时输出恰为全零波形）。④ 迭代中 `torch.istft` 与最终的 `torch.istft` 都硬编码 `center=True`，而 `torch.stft` 也硬编码 `center=True, pad_mode="reflect"`；当 `length=None` 时逆变换输出 `(T-1)*hop` 点，重建帧数恰为 `T`，但小 hop 下 reflect 填充会要求 `len(inverse) > n_fft//2`，否则报错。⑤ 给了 `length` 时帧数必须满足 `1 + length//hop == T`，否则 `specgram * angles` 广播失败；参考解用 `rebuilt frame count does not match specgram` 显式报错。

输出含 `out` / `shape` / `target_length` / `momentum_effective` / `n_iter` / `power` / `n_frames` / `envelope_min`。

## 8. 官方 quirk 清单

① reflect 填充排除边界样本；② `win_length < n_fft` 的零补左偏；③ Hermitian 镜像自动丢弃 DC/Nyquist 虚部；④ 窗包络除法与 `1e-11` NOLA 阈值；⑤ `length` 超长时补零而 envelope 补 1；⑥ `normalized` 在 stft / istft 方向相反；⑦ `True` 与 `'window'` 等价、`'frame_length'` 另走一支；⑧ `pad` 只在 `length` 给定时生效、`pad_mode` 从未被使用；⑨ momentum 先 `m/(1+m)`；⑩ `tprev` 存未归一化谱且 `mul_` 原地；⑪ `+1e-16` 归一化；⑫ `n_iter=0` 仍使用 `rand_init` 的初相位；⑬ 帧数与 `length` / `hop` 的约束。

## 9. 验证与落地

```text
probe860.py   : PROBE 860 DONE ok=105 fail=0
verify860.py  : 2960 checks ALL GREEN / 0 SKIP
driver860.py  : 45/45 ALL ACCEPTED
mut860.py     : 226 total / 4 equivalent / 222 active
                222 CAUGHT / 0 SURVIVED / 0 ERROR
selftest      : run + submit both ACCEPTED
precheck860.py: all PASS
```

`land860.py` 落地 `problems/pytorch/au-torchaudio-griffinlim-phase-ledger/` 共 11 文件，三件套与 `variants/pytorch/` 用 `filecmp` 校验一致。
