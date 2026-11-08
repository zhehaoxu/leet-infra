# torchaudio functional 波形算子账：mu-law 压扩与 int64 截断、convolve 的操作数交换与原始长度切片、fftconvolve 的 DFT 乘积、_apply_convolve_mode 的中心裁剪、add_noise 的 SNR 缩放与 lengths quirk、phase_vocoder 的相位累加与极坐标重构

## 1. 背景

本题重建 `torchaudio.functional` 中**波形侧**九个顶层函数的纯 Python 数值账，而不是调用库函数。判题环境没有 `torchaudio`，也不允许依赖 `torch` / `numpy`；所有 mode 必须是纯 Python 的确定性数值账，输入是嵌套列表，输出是题目约定的复合字典。与 order 855 的「频谱后处理账」划界：855 只管幅度谱 / DB / delta / CMN / spectrogram，本题只管 `mu_law_encoding` / `mu_law_decoding` / `convolve` / `fftconvolve` / `_check_shape_compatible` / `_check_convolve_mode` / `_apply_convolve_mode` / `add_noise` / `phase_vocoder` 这九段波形算子。

### 官方源码位置

回源目标是 `pytorch/audio main::src/torchaudio/functional/functional.py`（2511 行，order 855 已回源并落盘在 `.workbuddy/scratch/captum855/functional.py`）。九个函数在文件第 668–735 行（mu-law 一对）、第 2189–2315 行（卷积族）、第 2317 行起（`add_noise`）、第 732–802 行（`phase_vocoder`）。源码一行不改，用 `ast.get_source_segment` 按函数名整段抽取（含签名、docstring、注释）。

### 逐字执行环境

九个函数 `exec` 到同一个共享命名空间 `NS`；`NS` 只注入 `math` / `typing` / `warnings` / `torch` 与若干 typing 别名，不 mock 任何官方函数体。驱动层统一把嵌套列表转成 `float64` 张量（复数用 `complex128`），因此题目里的所有数值常数都是 IEEE double 口径；`mu_law_decoding` 对非浮点输入会 `to(torch.float)` 隐式降到 float32，本判题链路显式把 int64 编码值重新包成 `float64` 张量再解码，避免这条隐式降精度路径成为判题数值来源。

## 2. 口径约定

### 输入与输出

单入口是 `solve(mode, **kw)`，六个 mode 的必填参数与默认值如下。`conv_mode` 对应官方的 `mode` 形参（避免与 `solve` 的分派名冲突）：

```text
solve("mu_law",        x, quantization_channels)
solve("convolve",      x, y, conv_mode="full")
solve("fftconvolve",   x, y, conv_mode="full")
solve("slice_mode",    full, x_length, y_length, conv_mode="full")
solve("add_noise",     waveform, noise, snr, lengths=None)
solve("phase_vocoder", real, imag, rate, phase_advance)
```

### 数值精度与非法输入

全部用 Python `int` / `float`，嵌套列表按 C 序展平与回填；广播规则与 NumPy 一致：从尾部对齐，每维必须相等或其中之一为 1。不使用全局随机源，所有波形、噪声、SNR、相位推进量显式入参。未知 mode 抛 `ValueError` 且消息含 `unknown mode`；所有非法输入抛 `ValueError`，消息点名被违反的约束（例如 `quantization_channels must be an integer`、`The operands must be the same dimension`、`rate must be a positive number`）。校验顺序本身也是契约的一部分：两个输入同时非法时，以先校验的那个为准。

## 3. mode 1：mu_law

### 编码与解码公式

设 `mu = quantization_channels - 1`、`L = log1p(mu)`。编码是 `sign(x) * log1p(mu*|x|) / L`，再线性搬到 `[0, mu]` 并加 `0.5` 后截断成 `int64`；官方 `.to(torch.int64)` 是**向零截断**，而搬移后的值非负，因此 `+0.5` 等价于 round-half-up。解码把整数码 `e` 映回 `[-1,1]`：

```text
mu  = qc - 1
L   = log1p(mu) = log(1 + mu)
x_mu = copysign(log1p(mu * abs(x)) / L, x)
e    = int((x_mu + 1) / 2 * mu + 0.5)
dx   = (e / mu) * 2 - 1
dec  = copysign((exp(abs(dx) * L) - 1) / mu, dx)
```

端点性质：`x=-1 / 0 / 1` 分别编码成 `0 / qc//2 / qc-1`；中点 0 解码回来是一个极小的正数（`qc=256` 时约 `8.62e-05`）而不是精确 0；`x=0.3, qc=256` 编码为 228、解码为 `0.30633397981616217`。编码结果保持输入的嵌套形状，元素类型是 Python `int`。

### 域约束

`quantization_channels` 必须是 `>= 2` 的整数，且 `bool` 不算整数（规避 `isinstance(True, int)` 陷阱）；`x` 必须是非空数值数组且每个样本落在 `[-1, 1]`。违反分别抛含 `quantization_channels must be an integer` / `at least 2` / `x must be scaled to between -1 and 1` / `x must be a non-empty numeric array` 的 `ValueError`。

## 4. mode 2：convolve

### 直接卷积与操作数交换

官方用 `torch.nn.functional.conv1d` 加 `weight.flip(-1)` 实现**真卷积**（不是互相关）。`full` 结果长度恒为 `N + M - 1`。★ 当 `x` 的尾维短于 `y` 时，官方会交换两个操作数再做卷积（真卷积可交换，数值不变），但最后 `_apply_convolve_mode(result, x_size, y_size, mode)` 传入的仍是**交换前的原始长度**；因此 `same` 的长度是原始 `x_length`，`valid` 的长度是 `max-min+1`，都不会因为交换而改用 `y` 的长度。`swapped` 字段必须如实记录是否发生了交换，等长时不交换。

```text
full = [ sum_i a[i] * b[k-i] for k in 0..N+M-2 ]   (0 <= k-i < M)
valid: target = max(N, M) - min(N, M) + 1 ; start = (N+M-1 - target) // 2
same : target = N (original x_length)      ; start = (N+M-1 - N) // 2
out  = full[..., start : start + target]
```

### 前导维广播

两操作数必须同 rank，且除尾维外的每个前导维相等或其中之一为 1，否则抛 `The operands must be the same dimension` 或 `Leading dimensions of x and y are not broadcastable`。合法时把前导维广播成统一形状 `lead`，展平成 `B = prod(lead)` 行，逐行做一维卷积，再回填成 `lead + [L]`。返回值还给出 `num_signals = max(1, prod(lead))` 与 `leading_shape`。

## 5. mode 3：fftconvolve

`fftconvolve` 在 `n = N + M - 1` 点上做 `rfft(x) * rfft(y)` 再 `irfft`，与直接法数值一致（浮点误差在 `1e-8` 内）。★ 它**不交换**操作数：即使 `x` 更短也保持原顺序，`swapped` 恒为 `False`。切片规则与直接法完全相同（同样使用原始 `x_length` / `y_length`）。本题用纯 Python 的 DFT / IDFT 复现这一路径，缺失的高频采样等价于补零，因此 `irfft` 的输出必须是实数 `float`，而不是虚部约等于 0 的复数。

## 6. mode 4：slice_mode

把 `_apply_convolve_mode` 单独抽出来判题：给定显式 `full` 结果与两个原始长度，返回中心裁剪后的 `out`，并给出 `start_idx` / `target_length`。`full` 直接原样返回；`valid` 的长度是 `max-min+1` 而不是 `|N-M|` 的其它变体；`same` 的长度恒为 `x_length`（即使 `x` 是较短的操作数）。中心化用整除 `// 2`，full 长度为奇数时起点**偏左**。多维 `full` 时只对最后一维切片，且递归时起点必须原样传递。

## 7. mode 5：add_noise

### 能量与 scale

`waveform`、`noise` 的最后一维长度必须相同，`snr`（以及可选的 `lengths`）的 rank 必须等于 `waveform` 的前导维 rank；前导维按广播规则统一成 `lead`。对每一行 `b`：

```text
Es = sum_i waveform[b][i]^2   (i < length[b])
En = sum_i noise[b][i]^2      (i < length[b])
original = 10 * (log10(Es) - log10(En))
scale    = 10 ** ((original - snr[b]) / 20)
out[b]   = waveform[b] + scale * noise[b]
```

等价地 `scale = sqrt(Es/En) * 10^(-snr/20)`；SNR 每升高 20 dB，scale 缩小 10 倍。`Es` 或 `En` 非正时抛 `energy_signal and energy_noise must be positive`。

### lengths 的官方 quirk

★ `lengths` 只用于**计算能量**（把超出长度的样本当 0），相加时官方仍用 `scale.unsqueeze(-1) * noise` 与**未 mask 的原始 noise**、以及未 mask 的原始 `waveform`。因此 padding 区域虽然不贡献能量，却照样被加上了缩放噪声。例：`waveform=[1,2,3,100]`、`noise=[1,1,1,1]`、`snr=0`、`lengths=3` 时 `Es=14`、`En=3`、`scale=sqrt(14/3)`，第 4 个输出是 `100 + scale` 而不是 `100`。此外 `snr` / `lengths` 的前导维若与波形冲突，必须显式拒绝，不能静默扩维。

## 8. mode 6：phase_vocoder

### 时间轴与相位累加

`rate == 1.0` 时官方**直接返回输入**，不做极坐标往返，因此 `real` / `imag` 是输入原对象、帧数不变。其它情况：

```text
time_steps = arange(0, T, rate)          -> n = ceil(T / rate), t_j = j * rate
alphas     = time_steps % 1.0
phase_0    = angle(spec[..., :1])
padded     = pad(spec, [0, 2])
s0, s1     = padded[..., int(t)], padded[..., int(t) + 1]
phase      = angle(s1) - angle(s0) - phase_advance
phase      = phase - 2*pi*round(phase / (2*pi))     # wrap into [-pi, pi]
phase      = phase + phase_advance
phase      = cat([phase_0, phase[..., :-1]])         # shift by one frame
phase_acc  = cumsum(phase)
mag        = alphas * |s1| + (1 - alphas) * |s0|
out        = polar(mag, phase_acc)                   # cos / sin reconstruction
```

`phase_advance` 的形状是 `(freq, 1)`，本题按每个频点一个标量展开；`real` / `imag` 必须同形状且至少 2 维 `(..., freq, frames)`。输出帧数 `num_frames_out = n`，并原样给出 `time_steps` / `alphas` / `num_frames_in` / `freq` / `rate`。`rate <= 0`、非数值 `rate`、频点数与 `phase_advance` 不符都要抛带关键字的 `ValueError`。

### 域约束

`real` / `imag` 不同形抛 `real and imag must have the same shape`；rank 不足抛 `complex_specgrams must have (..., freq, frames) shape`；`rate` 非法抛 `rate must be a positive number`；`phase_advance` 数量不符抛 `phase_advance must have one value per freq bin`。批量前导维会被展平成 `B` 组独立处理，最后回填成 `lead + [freq, n]`。

## 9. 返回字段一览

```text
mu_law        : encoded, decoded, mu, quantization_channels, shape
convolve      : out, shape, full, full_length, mode, swapped,
                x_length, y_length, leading_shape, num_signals
fftconvolve   : (same fields as convolve, with swapped always False)
slice_mode    : out, shape, mode, start_idx, target_length
add_noise     : out, shape, scale, energy_signal, energy_noise,
                leading_shape, length
phase_vocoder : real, imag, shape, time_steps, alphas,
                num_frames_in, num_frames_out, freq, rate
```

## 10. 自测与判题

判题用例是 20 个顶层 `check_*(solve)`，覆盖六个 mode 的数值、形状、返回字段、域约束、校验顺序、三维非对称前导广播、单例前导维的末维取值、等长操作数不交换、三抽头核、中心裁剪正起点、`y_length=1`、SNR 前导维冲突、bool 数组元素、FFT 输出实数类型、以及相位声码器的实部/虚部数值。所有数值取自官方逐字执行环境并与参考解、独立 oracle 三层对拍。
