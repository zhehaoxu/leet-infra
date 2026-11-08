# 855 解答笔记：torchaudio 频谱后处理账

## 1. 官方回源

### 源码清单与依赖

主源码是 `pytorch/audio main::src/torchaudio/functional/functional.py`（2511 行 / 95067 bytes）。与 Captum 不同，torchaudio 的 owner 仍是 `pytorch`，未迁移到 `meta-pytorch`；用 `api.github.com/repos/pytorch/audio/contents/...?ref=main` 取 base64 内容即可。判题环境没有 torchaudio，所以不能 `import torchaudio.functional`；官方文件顶部还有 `.filtering` 的相对导入，直接整文件 `exec` 会失败。

### 逐字执行环境

解法与 852 的 `create_dct` 抽取同款：用 `ast.get_source_segment` 抽出七个目标函数的逐字源码，注入 `torch` / `math` / `warnings` / `Tensor` / typing 后 exec 到共享 `NS`。

```text
WANT = ["spectrogram", "_get_spec_norms", "_get_complex_dtype",
        "amplitude_to_DB", "DB_to_amplitude",
        "compute_deltas", "sliding_window_cmn"]
NS = {"math": math, "warnings": warnings, "torch": torch,
      "Tensor": torch.Tensor, "Optional": ..., "Union": ...}
for name in WANT:
    seg = ast.get_source_segment(SRC, node)
    exec(compile(seg, "functional.py::" + name, "exec"), NS)
```

驱动层把嵌套列表转 `float64` 张量；`run_spectrogram` 临时把 `torch.stft` 换成 spy 记录精确 kwargs，finally 恢复；复数结果统一拆成 `{"real": ..., "imag": ...}`。全局默认 dtype 不改动，因为 `float64` 输入已经让 `stft` 走 `cdouble`。

## 2. 十二段官方机制探针

### 探针结论

12 段探针共 87 项断言全绿。要点：`_get_spec_norms` 四种合法输入与两类异常；`amplitude_to_DB` 的 10/20 倍率、`amin` clamp、`db_multiplier` 减法、4D per-batch cutoff 与 3D/2D 单 cutoff；`DB_to_amplitude` 的 `power=1/0.5` 往返；`compute_deltas` 的 `n`、`denom`、互相关符号、replicate/constant 边界差异、3D pack/unpack；CMN 的窗口调度、朴素窗口均值等价、`norm_vars`、`window_frames==1` 置零、2D 降维；`_get_complex_dtype` 映射；`spectrogram` 的 stft kwargs、`pad` constant padding、`return_complex` 警告、归一化守卫；以及用手工 DFT 对拍帧构造、窗口居中补零、center reflect、`frame_length`/`window` 两种缩放、power 分支与 `onesided=False`。

## 3. 三层对拍

### official 层

逐字执行的官方函数，覆盖 2D/3D/4D/5D 的 dB、1D/2D 的 dB 逆变换、W=3/4/5/7 的 deltas、7 组 CMN 参数、13 组 spectrogram 参数与 2D waveform。

### reference 层

`solution_staged.py` 的五个 mode，全部纯 Python，无 torch 依赖；`_shape` / `_flatten` / `_reshape` 负责任意维度展平与回填。

### oracle 层

独立实现：dB 用逐 batch 显式 `log10`；deltas 用 `sum_i i*(c[t+i]-c[t-i]) / (2*sum i^2)`；CMN 用朴素窗口均值与方差；spectrogram 用 `cmath.exp(-2*pi*i*k*j/n_fft)` 的 DFT。三层互不相同，避免同源错误。

### 对拍结论

```text
total checks: 1520    failures: 0
RESULT: ALL GREEN
```

首轮仅 2 项失败，是把 `torch.cfloat` 的字符串别名写成 `torch.complex64` 造成的断言写法问题；改成直接比较 dtype 对象后全绿。另一次探针失败来自把官方输出的 `[freq][time]` 误读成 `[time][freq]`，修正索引后 12 段全绿。

## 4. 官方 quirk 清单

### CMN 切片笔误

官方首帧 `input_part = specgram[:, window_start : window_end - window_start, :]` 的上界应是 `window_end`。但该分支只在 `last_window_start == -1` 时执行，即 `t=0`；而首帧的 `window_start` 无论 center 与否都被夹到 0，所以 `window_end - window_start == window_end`，笔误**不可观测**。探针枚举了 center/非 center、多种 `cmn_window` 与帧数，首帧起点恒为 0。参考解保留逐字形式（`range(ws, we - ws)`），mutation 中对应变异登记为真等价。

### 2D squeeze 降维

末尾 `if len(input_shape) == 2: cmn_specgram = cmn_specgram.squeeze(0)` 对 `(1, F)` 输出会压成 `(F,)`。因此 2D 输入且 `num_frames == 1` 时，输出是 1D 列表而不是二维；3D 输入不受影响。

### docstring 维度矛盾

`sliding_window_cmn` 的 docstring 把输入写成 `(..., time, freq)`、输出写成 `(..., freq, time)`，但源码用 `num_frames, num_feats = input_shape[-2:]` 且最后 reshape 回 `input_shape[:-2] + (num_frames, num_feats)`，形状与输入完全相同。

## 5. 五个 mode 的实现要点

### db

先算 `vals`，再按 `packed_channels * freq * time` 分块求每个 batch 的 cutoff 并逐元素 `max`。关键点：`num_batches = prod(shape[:-3])` 只在维度大于 3 时才大于 1；3D/2D 恒为 1。`cutoffs` 在 `top_db=None` 时必须是 `None`。

### db_inverse

`ref * (10^(0.1*v))^power` 逐元素映射，允许 1D 输入；返回值回显 `ref` 与 `power`。与 `db` 的 10/power=1、20/power=0.5 两种配对可往返。

### deltas

把全部前导维度展平成通道，逐通道 replicate/constant 填充后做 `kernel[j]*padded[t+j]` 互相关。偶数 `win_length` 必须用 `(W-1)//2`；`denom` 用官方闭式；元数据 `kernel` 是 `[-n..n]`。

### cmn

先算完整 `windows` 列表，再按首帧一次性累加、后续按 `last_window_start` / `last_window_end` 增减的递推维护 `cur_sum` / `cur_sumsq`。`norm_vars` 分支在原行上乘 `variance^-0.5`，`window_frames==1` 直接置零。末尾复现 2D 单帧 squeeze。

### spectrogram

顺序是 `pad` → `center` 补边 → 窗口居中补零 → 逐帧乘窗 → DFT → `frame_norm` 缩放 → `window_norm` 缩放 → power 分支。`window_norm` 用原始窗口平方和；`frame_norm` 除 `sqrt(n_fft)`。复数输出布局 `[freq][time]`，2D waveform 额外叠加 channel 维。

## 6. 域约束与错误消息

所有非法输入抛 `ValueError`，消息点名约束：`amin must be a positive number`、`top_db must be a number or None`、`at least two dimensions`、`must be rectangular`、`must contain only numbers`、`Window length should be greater than or equal to 3`、`win_length must be an int`、`mode must be replicate or constant`、`cmn_window must be a positive int`、`norm_vars must be a bool`、`waveform must be 1D or 2D`、`window length must equal win_length`、`Invalid normalized parameter`、`Input type not supported`（`TypeError`）、`pad_mode must be reflect or constant`、`window energy must be positive`、`reflect pad requires`、`waveform too short for n_fft`、`unknown mode`。

## 7. 判题用例设计

8 个 `check_*(solve)`：`check_db`（含 4D/5D per-batch cutoff 与 `packed_channels>1`）、`check_db_inverse`（往返）、`check_deltas`（偶数 W、符号、填充差异、3D）、`check_cmn`（7 组参数、center 与非 center 差异、2D 单帧降维、norm_vars）、`check_spectrogram`（13 组参数、2D waveform、pad 使 bins≠frames）、`check_domain`（约 35 条精确消息断言）、`check_metadata`（元数据字段逐项）、`check_unknown_mode`。driver 模拟 judge 注入 `solve`，参考解 8/8。

## 8. mutation 覆盖

```text
202 raw mutants, 5 true equivalents removed, 197 active: 197 CAUGHT, 0 SURVIVED, 0 ERROR
ALL MUTANTS CAUGHT
```

首轮 168 CAUGHT / 27 SURVIVED / 6 ANCHOR-ERROR。SURVIVED 主要是删掉域约束后没有对应非法输入，补约 20 条精确消息断言；另有 3 处「不可观测」需专门设计用例：4D `packed_channels>1`（原用例通道数恒为 1）、偶数 `win_length`（奇数时两种取整相同）、2D waveform + `pad=2` 使 bins≠frames。6 个 ANCHOR-ERROR 是锚点重复（`ws = t - cmn_window` 是 `ws = t - cmn_window // 2` 的前缀、8 空格缩进命中 12 空格行、两处 `center` 检查、首帧与增量分支同款累加行），全部改成带上下文的唯一锚点。

5 个真等价：`cmn-slice`（首帧起点恒 0）、`h-reshape-scalar`（标量分支在本题输入域内不可达）、`h-rows-cast`（`float(x)` 对数值无影响）、`sp-cos-sign`（cos 是偶函数）、`sp-power1`（`abs(z)^1 == abs(z)`）。

## 9. 常见踩坑

- 官方复数输出是 `[freq][time]`，按 `[time][freq]` 比较会全盘错位。
- reflect 左段从 `x[p]` 开始、右段从 `x[-2]` 开始；写成 `x[p-1]` / `x[-1]` 只在首帧错，容易漏测。
- CMN 的 `min_cmn_window` 判断条件是 `window_end > t`，不是「窗口帧数小于最小值」。
- `normalize` 的布尔值判定要先处理 `str` 再处理 `bool`，否则 `True`（bool 是 int 子类）会走错分支。
- deltas 的 `pad_mode` 与外部 `solve(mode, ...)` 同名冲突，必须在 `solve` 里改名映射。

## 10. 自检清单

官方 12 段探针全绿、三层对拍 1520 项 ALL GREEN、driver 8/8、mutation 197/197 CAUGHT，随后才是四文档中英版、meta、starter 预检、11 文件落地、selftest 双模式、全库双语校验、collections 注册、todos 更新、短护栏与全量护栏。
