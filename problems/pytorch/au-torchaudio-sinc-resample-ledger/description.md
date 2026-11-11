# torchaudio sinc 重采样账：GCD 约分与 width、 Hann/Kaiser 窗与 beta 的 float32 降级、stride=orig_freq 的相位交错卷积、以及弃用别名落错分支的官方 quirk

## 1. 背景

`torchaudio.functional.resample` 用带限 sinc 插值重采样波形。本题把整条链路拆成五个可独立判题的纯 Python 数值账：参数约分账、修正 Bessel I0 级数、核矩阵构造、相位交错卷积、以及端到端编排。
本题与 `audio-resample-polyphase`（344，整数比多相重采样）按算法族划界：那里是多相滤波器组与整数抽取/插值，这里是 sinc 核 + `F.conv1d(stride=orig_freq)` 的相位矩阵。与 855 / 857 / 858 虽同源于 `functional.py`，但函数集合互不重叠。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _get_sinc_resample_kernel    (lines ~1303-1400)
  _apply_sinc_resample_kernel (lines ~1403-1427)
  resample                    (lines ~1430-1490)
```

### 逐字执行环境

暂存文件 `.workbuddy/scratch/torchaudio859/official859.py` 用 `ast.get_source_segment` 从 855 已落盘的 2511 行官方文件中整段抽取上述三个顶层函数，`exec` 到共享命名空间；只注入 `math` / `typing` / `warnings` / `torch` / `Tensor` / `Optional` / `_CPU`，不 mock 任何函数体。驱动层统一 float64。

## 2. 口径约定

### 输入与输出

```text
solve(mode, **kw) -> dict
mode = meta | i0 | kernel | apply | resample
```

参考解暴露单入口 `def solve(mode, **kw)`；判题文件顶层是一组 `def check_*(solve)`，由 judge 注入 `solve`。判题用例不得 `from solution import ...`。
全部输入是嵌套 Python 列表，纯 Python 实现，不依赖 `torch` / `torchaudio` / `numpy`，不使用全局 RNG。任意前导维按 C 序展平成等长行，逐行计算后按 `lead + [tail]` 折回。

### 通道维与数值精度

官方 kernel 张量形状是 `(new_freq, 1, 2*width+orig_freq)`，中间那一维恒为 1。★ 判题口径把它剥掉：`kernel` / `apply` 的核矩阵统一是二维 `(n, K)`，而 `meta.shape` 仍保留官方三维形状。
判题数值全程 float64，对应官方 `resample` 把 `waveform.dtype` 传进核构造器的路径。官方 `dtype=None` 时 `idx` 是 float64 而 `torch.arange(0,-n,-1,dtype=None)/n` 由 int64 除法得 float32，最后 `kernels.to(torch.float32)` 整体降级；这三条 dtype 路径互不相等（实测最大差约 1.4e-08），本题不把它们纳入判题域。
Kaiser 分支的 `beta_tensor = torch.tensor(float(beta))` 恒为 float32，分母 `torch.i0(beta_tensor)` 是 float32 而分子是 float64。判题口径用「float64 精确级数再 f32 量化」复刻分母；`torch.i0(float32)` 的 Cephes 多项式在部分 beta 上有几 ULP 偏差，故 Kaiser 判题域只取默认 beta 或逐位一致的值（如 3.0、12.0）。

### 非法输入

所有非法输入抛带具体关键字的 `ValueError`：`must be a non-empty numeric array` / `Frequencies must be of integer type` / `must be an integer` / `must be at least` / `must be positive` / `beta must be a number` / `Invalid resampling method` / `kernels must be a 2-D` / `kernel shape must be` / `should be positive` / `unknown mode`。校验顺序与精确消息都属于契约。

## 3. mode 1：meta

### 约分与核尺寸

```text
g = gcd(orig_freq, new_freq)
o = orig_freq // g ; n = new_freq // g
base_freq = min(o, n) * rolloff
width     = ceil(lowpass_filter_width * o / base_freq)
K         = 2 * width + o
shape     = [n, 1, K]        # official tensor
scale     = base_freq / o
```

注意 `width` 用的是约分后的 **orig_freq**（`o`），`base_freq` 用的是 `min(o, n)`；降采样 8→4 时 `min(2,1)=1`，所以 `base_freq` 等于 rolloff 本身。
`meta` 同时暴露 `method` / `deprecated` / `replacement` / `window_branch` / `warnings` / `beta` / `beta_default_used` / `scale`，其中 `beta` 是 `f32` 量化后的值（默认 14.769656459379492 → 14.76965618133545）。

### 弃用别名 quirk

★ 官方对 `sinc_interpolation` / `kaiser_window` 只发一个 deprecation warning，**从不改写局部变量** `resampling_method`；窗口分支的判据是字面量 `resampling_method == "sinc_interp_hann"`，因此除这个字面量以外的所有名字都落进 else（Kaiser）分支。也就是说 `sinc_interpolation` 得到的是 Kaiser 核，尽管警告声称 "The default behavior remains unchanged" 并指向 `sinc_interp_hann`。本题如实复刻。

## 4. mode 2：i0

### 修正 Bessel 级数

```text
I0(x) = sum_{k=0}^{inf} (x^2 / 4)^k / (k!)^2
term_0 = 1 ; term_k = term_{k-1} * (x^2/4) / k^2
i0          -> float64 series
i0_float32  -> f32(float64 series)
i0_beta_float32 = f32(I0(f32(beta_default))) = 271903.5625
```

级数对判题域内 x ≤ 20 收敛到约 1e-16 相对精度，与 `torch.i0(float64)` 逐值一致。`values` 可为标量或嵌套列表；`i0(0)=1` 且 I0 是偶函数。

## 5. mode 3：kernel

### t 网格与 clamp

```text
idx[k] = k / o                  for k in [-width, width + o)
t[j,k] = (-j / n + idx[k]) * base_freq
t      = clamp(t, -W, W)        W = lowpass_filter_width
t     *= pi                     # after the window is evaluated
```

### 窗口、sinc 与 scale

```text
hann   : window = cos(t * pi / W / 2) ** 2
kaiser : window = I0(beta * sqrt(1 - (t / W) ** 2)) / I0(beta)
sinc   = 1.0 if t*pi == 0 else sin(t*pi) / (t*pi)
kernel[j,k] = sinc[j,k] * window[j,k] * (base_freq / o)
```

`kernel` 返回二维 `kernels`（`[n, K]`）以及账本视图 `t` / `window` / `sinc`，外加 `width` / `kernel_length` / `scale` / `beta_float32` / `i0_beta_float32` / `meta`。
★ clamp 到 ±W 时 Hann 窗不是精确 0，而是 `cos(pi/2)**2 ≈ 3.7e-33` 的浮点残差；Kaiser 窗在边界是 `1/I0(beta) ≈ 3.7e-6`，明显非零。两者都能在核矩阵首尾观察到。

## 6. mode 4：apply

### 补零、stride 卷积与相位交错

```text
padded      = [0] * width + waveform + [0] * (width + o)
num_frames  = length // o + 1
phase[j][s] = sum_k padded[s*o + k] * kernels[j][k]
flat        = [phase[j][s] for s in frames for j in phases]
target_len  = ceil(n * length / o)
out         = flat[:target_len]
```

`F.conv1d` 是互相关而非真卷积，不翻转核；输出 `(B, n, num_frames)` 先 `transpose(1,2).reshape(B,-1)` 交错相位，再按 `target_length` 截断，最后按输入前导维 `view` 回去。
`apply` 返回 `out` / `shape` / `target_length` / `num_frames` / `conv_length` / `padded_length` / `width` / `stride`；核矩阵形状必须是 `[n, 2*width+o]`。

## 7. mode 5：resample

`resample` 先做 `orig_freq <= 0 or new_freq <= 0` 的拒绝（这个 guard 只在编排层，核构造器没有），再在 `orig_freq == new_freq` 时**原样返回输入**；否则按同一组参数构造核并调用 `apply`，返回 `out` / `shape` / `target_length` / `identity` / `meta`。

## 8. 验证结果

```text
PROBE  859 DONE ok=124 fail=0
VERIFY 859 DONE checks=727 fails=0 ALL GREEN
DRIVER 859 35/35 ALL ACCEPTED
MUT    859 196 total / 8 equivalent / 188 active / 188 CAUGHT
       0 SURVIVED, 0 ERROR, ALL MUTANTS CAUGHT
```

首轮变异有 25 项 SURVIVED 与 1 项锚点索引 ERROR；追加 6 组硬化用例（默认值、精确错误消息、bool/字符串类型、sinc 列、恒等短路的 meta）并判定 8 项真等价后全部收口。

## 9. 官方 quirk 清单

① 弃用别名只警告不改名，`sinc_interpolation` 实际走 Kaiser；② `beta_tensor` 恒 float32，分母是 float32 的 `i0` 而分子是 float64；③ Hann 在 clamp 边界不是精确 0；④ Kaiser 在 clamp 边界是 `1/i0(beta)` 而非 0；⑤ `dtype=None` 时 `arange` 是 int64、除法降 float32、`idx` 却是 float64，最终核整体降 float32；⑥ `orig_freq == new_freq` 直接返回原张量对象；⑦ `<=0` guard 只在 `resample`；⑧ `target_length` 用 `ceil(n*L/o)` 而非卷积输出长度；⑨ 警告文本声称 "default behavior remains unchanged"，与①矛盾。

## 10. 查重与复用

落地前扫描确认 `max_order = 858`、859 空闲、836 题 order 唯一、无同名 slug；全库 `meta.json` 检索 `sinc_resample` / `sinc_interp_kaiser` / `griffinlim` / `spectral_centroid` 零命中。与 344 按算法族划界，与 855（频谱后处理）、857（波形算子）、858（NCCF 音高）虽同源于 `functional.py`，但函数集合互不重叠。
