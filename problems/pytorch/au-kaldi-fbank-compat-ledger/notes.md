# 852 解答笔记：Kaldi 兼容前端口径账

## 1. 官方回源

### 两份源码与 create_dct

从 `pytorch/audio` 的 `main` 分支拉取 `src/torchaudio/compliance/kaldi.py`（813 行）与 `src/torchaudio/functional/functional.py`。`kaldi.py` 的 `mfcc` 依赖 `torchaudio.functional.create_dct`，而判题环境没有该包，因此用 AST 从官方 `functional.py` 逐字抽取 `create_dct` 单独存成 `create_dct.py`，避免手写 DCT 引入偏差。

### 逐字执行环境

新建 `official_kaldi.py`：注入 fake `torchaudio` 与 `torchaudio.functional`，把官方抽取的 `create_dct` 填进 `_DCT_NS` 后挂到 fake 模块，再逐字 `exec` `kaldi.py`。冒烟确认命名空间有 36 个符号、五种窗类型齐全、`create_dct(4,4)` 首列全为 `0.5`、mel 与 inverse 互逆。探针跑的是真正的官方实现。

## 2. 十二段官方机制探针

### 探针结论

`probe852.py` 十二段全部跑通，钉死：五种窗函数与 `povey = hann^0.85`、blackman 在 `0.42` 等于 `torch.blackman_window` 而 `0.5` 不等；`snip_edges` 两种分帧公式与反射填充、too short 返回 `(0,0)`；`_get_log_energy` 的 epsilon 恒为 `1.1920928955078125e-07` 且 floor 在 log 域；preemphasis 的 replicate 首样本；`raw_energy=True/False` 的能量计算时机；dither 到 post energy 的完整链顺序；mel 的 `1127/700` 口径；`mel_freq_delta` 除以 `num_bins+1` 与 `center_freqs` 的 `(num_bins,1)` 形状；VTLN 四次赋值覆盖顺序；warp 不为 1 时三列 mel 全弯折；`spectrogram` 第 0 列被能量覆盖；fbank / mfcc 的开关与 `htk_compat` 尾部编排。

```text
WINDOW_TYPES = ['hamming', 'hanning', 'povey', 'rectangular', 'blackman']
```

## 3. 三层对拍

### official 层

`official_kaldi.py` 逐字执行官方 `_feature_window_function` / `_get_strided` / `_get_window` / `get_mel_banks` / `vtln_warp_freq` / `fbank` / `mfcc`，输入用 float64 张量。

### reference 层

即本题参考解 `solution.py`，五个 mode 全部走 `solve(mode, **kw)`，纯 Python float64。

### oracle 层

`verify852.py` 独立重写纯 Python 公式，不 import 参考解的任何函数：DFT 用 `cmath.exp` 的朴素累加、mel 与 inverse、三角滤波器组、VTLN 三段、前处理链、DCT 与 lifter 全部独立实现。

### 对拍结论

五个 mode 共 22556 个检查项，最终 ALL GREEN。official-vs-reference 默认 `tol=1e-6, rel=1e-5`，mfcc `features` 单独 `tol=5e-5, rel=0.0`；oracle-vs-reference 始终 `1e-7`。新增 `pad-negative` 用例覆盖 `pad <= 0` 分支。

## 4. 官方 quirk 清单

### center_freqs 形状

官方 docstring 声称 `center_freqs` 是 `(num_bins,)`，冒烟时 `round()` 一个列表直接抛 `TypeError: type list doesn't define __round__ method`，实测返回是 `(num_bins, 1)`；参考解保留双层列表。

### float32 量化误差

`get_mel_banks` 内部 `torch.arange(int64) * float` 降为 float32，`center_freqs` 几千 Hz 处约 `1e-3` 量化误差，经 DCT 累积到 mfcc `features` 约 `2.5e-5` 绝对误差。首轮对拍 241 failures 全在 `OFF mb*.center_freqs`，改用相对容差后降到 57，再对 mfcc 单独放宽才全绿。

```text
chk_off(tag, got, exp, tol=1e-6, rel=1e-5)
chk_off("mf%d.features" % ci, ref["features"], off, tol=5e-5, rel=0.0)
```

### VTLN 覆盖顺序

官方注释 `order of operations matter here`，四次赋值 `after_h -> before_h -> before_l -> outside_low_high_freq` 靠重叠区间覆盖，区间外频率原样保留。

### blackman 系数

默认 `blackman_coeff=0.42` 时官方手写公式恰等于 `torch.blackman_window`；探针初版把断言写反，实测后修正为「`0.42` 相等、`0.5` 不等」。

## 5. 五个 mode 的实现要点

### vtln_warp

先算 `l/h/scale/Fl/Fh/scale_left/scale_right`，再按四次赋值顺序写 `res`；`_warp_mel_py` 把 mel 列取 inverse mel、弯折、再取 mel 包回。

### mel_banks

`mel_freq_delta` 分母 `num_bins+1`；warp=1 用 `max(0, min(up,down))`，warp!=1 用 mask 分支；`center_freqs` 保持 `(num_bins,1)`；`high_freq<=0` 与 `vtln_high<0` 各加一次 nyquist。

### framing

`_get_strided` 两种分帧公式与反射填充；`_get_window_py` 按固定顺序跑前处理链，`raw_energy` 决定能量时机；`_FRAMING_DEFAULTS` 提供默认值；`padded_window_size` 必须是偶数且不小于窗长。

### fbank

`bins` 右侧补零列、逐帧 `_rfft_abs`、`use_power` 平方、`use_log_fbank` 取 log、`use_energy` 按 `htk_compat` 决定能量列位置、`subtract_mean` 做逐列 CMS。

### mfcc

强制 `True, True, use_energy, htk_compat, False` 调 fbank；DCT 首列强制覆盖；lifter 的 C0 恒 `1.0` 且 `0.0` 时跳过；`htk_compat` 把 C0/energy 移到末尾。

```text
dct[:, 0] = sqrt(1 / num_mel_bins)
```

## 6. 域约束与错误消息

所有非法输入统一抛 `ValueError` 且消息点名被违反的约束：窗类型的 `Invalid window type`、`window_shift` 的 `must be positive`、`padded_window_size` 的 `must be >= window_size` 与 `must be even`、dither 的 `must be provided` 与 `shape mismatch`、VTLN 的 `vtln_low must be higher` / `vtln_high must be lower` / `must be positive` / `inflection points out of range`、`mel_banks` 的 `at least 3 mel bins` / `must be even` / `Bad values in options: low-freq / high-freq / nyquist` / `vtln-low / vtln-high`、mfcc 的 `num_ceps cannot be larger than num_mel_bins`、未知 mode 的 `unknown mode`。

```text
Bad values in options: vtln-low / vtln-high
```

## 7. 判题用例设计

### 独立 oracle

7 个 `check_*(solve)` 各自带按官方算式重写的纯 Python oracle，对每个返回字段逐一比对；波形与噪声用确定性 LCG 生成，不依赖随机源。判题用例不 `import solution`，`check_*` 必须接受 `solve` 参数。

### 双向断言

除等于 oracle 外还断言结果必须不等于「看起来也合理」的错误候选：HTK mel 口径、`num_bins` 分母、`max/min` vs mask 分支、零填充 vs 反射填充、能量域 clamp vs log 域、`warp=1` vs `warp!=1` 的 bins、`use_energy` 首/尾列、C0 覆写、`htk_compat` 尾部编排。

### 域约束用例

非法输入的形状必须合法，确保删掉目标检查后不会被下游形状校验顶替报错；同时断言消息里的关键字（`vtln-low` 而非宽泛的 `vtln`）。全零波形钉死默认 `energy_floor=1.0`，显式 blackman 分支钉死默认 `blackman_coeff=0.42`。

## 8. mutation 覆盖

`mut852.py` 原始 154 项变异，覆盖 mel / epsilon / log-energy、五种窗函数、`_get_strided` 分帧、`_get_window` 前处理链、VTLN 四次赋值与返回字段、`get_mel_banks` 全部算式与两条分支、DFT / CMS、fbank 开关、DCT / lifter / mfcc 编排、九个默认值与 `solve` 分派。首轮 146 CAUGHT / 7 SURVIVED / 1 ANCHOR-ERROR：4 项确认为真等价（`vt-before-l-cond`、`mb-mask-up-cond`、`mb-mask-down-cond`、`dct-ortho-zero`）登记剔除，3 项补齐判题缺口，1 项修锚点；接线后重跑 **有效 150 项：150 CAUGHT / 0 SURVIVED / 0 ERROR**。

## 9. 常见踩坑

- 判题环境没有 `torchaudio`，官方逐字执行必须注入 fake `create_dct`。
- 官方 float32 与参考解 float64 的 `center_freqs` 量化误差要放相对容差，mfcc 再单独放宽。
- `center_freqs` 是 `(num_bins,1)`，别在工具里 `round()` 嵌套列表。
- oracle DFT 用 `math.fsum` 累加复数会抛 `TypeError: must be real number, not complex`，改用 `cmath.exp` 朴素累加。
- `cepstral_lifter=0.0` 必须跳过 lifter，否则除零。
- 判题用例写 `check_*(solve)`，mutation runner 同步用 `getattr(T, name)(S.solve)`。
- 域约束只断言异常类型不够，必须断言消息关键字。
- 等价变异要保留记录但剔除统计，锚点必须唯一。

## 10. 自检清单

- 三层对拍 22556 项 ALL GREEN。
- 判题用例 7/7 通过参考解。
- mutation 有效 150 项全 CAUGHT。
- 双语校验零新增 error / warning。
- selftest run 与 submit 双模式全 ACCEPTED。
