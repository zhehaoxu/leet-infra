# 出题笔记：869 = `au-torchaudio-biquad-coeff-ledger`

## 1. 选题与划界

855 回源的 `functional.py`（2511 行 / 55 个顶层函数）已被 855–868 全部覆盖，因此换题源到 `pytorch/audio main::src/torchaudio/functional/filtering.py`（1702 行 / 28 个顶层函数，2026-11-10 经 GitHub raw 抓取落盘）。
查重结论：九个系数函数 + `_dB2Linear` 在 `problems` 全库 meta / description / notes **零命中**；但 **861 已完整覆盖** `biquad` / `lfilter` / `_lfilter` / `_lfilter_core_generic_loop` 与 K-weighting IIR 递推。故 869 只判系数推导，`lfilter` 作 spy 注入。已排除 `filtfilt` / `flanger` / `phaser` / `overdrive` / `dither` / `dcshift`（RNG / 波表 / 循环）、`gain` / `contrast`（过薄）、`vad`（与 `asr-fsmn-vad-endpoint` 重叠风险）。

## 2. 官方逐字环境

`official869.py` 用 `ast.get_source_segment` 抽 11 个函数进共享 NS，函数体一字未改；注入真实 `torch`(float64) / `math` / `warnings` / `typing` / `F`、`_IS_TORCHAUDIO_EXT_AVAILABLE=False`。唯一替身是 `lfilter` spy：记录 `waveform_shape` / `a_coeffs` / `b_coeffs` / `clamp` / `batching` / `dtype`，原样返回 waveform。无 RNG、无音频 IO、无 warning。
冒烟 `SMOKE 869 DONE`：lowpass / bass 系数、`bass a0_raw=1.0`、`call` 的 clamp/batching/dtype、6 组 presets、7 类 errors（deemph/riaa 非法率 ValueError，Q=0 / sr=0 不抛错）、`biquad_order` 系数顺序、`grid` 9 kind / 26 行全通。

## 3. 关键机制

`w0=2πf/sr`、`alpha=sin(w0)/2/Q`、`A=exp(gain/40*ln10)`；SoX 架式 `temp1/temp2/temp3`。**`bass` 六系数先全 `/a0` 再给 `biquad`（a0_raw=1.0），其余函数传未归一化系数**；`equalizer` 的 `alpha*A` / `alpha/A` 不对称；`band` 另走 `bw_Hz=f/Q`、`a2=exp(-2π·bw_Hz/sr)`、`noise=True` 的 `mult` 分支且 `b1=b2=0`；`bandpass` 有 `const_skirt_gain`；`deemph`(44100/48000) 与 `riaa`(4 率) 固定参数表 + 非法率 ValueError，RIAA 另用 1 kHz 频响实部/虚部算 `g` 归一；`biquad` 把 a_coeffs 放前、b_coeffs 放后。

## 4. 验证流水线

12 段机制探针 `probe869.py` **485/485 ALL GREEN**：每段用独立闭式推导重算 a/b，并用 z 域传递函数的代数性质（全通幅值、DC 单位增益、陷波、峰值增益 A²、低架 DC=A²、1 kHz 归一）交叉验证；同时钉死 census 的 14 条源码事实、presets 参数表、非法率消息、Q=0 / sr=0 / 负率 / band Q=0 的 IEEE 边界。
两层对拍 `probe_solution869.py` **91419 checks / 2161 configs / 0 bad**：官方 `comp` 与参考 `solve` 在 9 kind × 5 采样率 × 5 频率 × 5 Q、bass/equalizer × 5 增益、band × noise、bandpass × skirt、presets、biquad、边界与五种 waveform 形状上逐项深比较，异常类型与消息也对齐。
三层独立 oracle `verify869.py` **26832 checks / 475 configs / 5679 代数性质 / 0 fails**：三角函数改走 `cmath.exp(1j*x)`、`A` 改走 `10**(gain/40)`、`exp` 改走 `cmath.exp`、`sqrt` 改走 `x**0.5`、π 改走 `4*atan(1)`；RIAA 多项式用 `Fraction` 精确重算；再叠加结构交叉检查（call / presets / grid / errors / census / db2linear / biquad_order 与 coeffs 的一致性、census 全新 dict、unknown mode 消息）。

## 5. 变异与等价留证

`mut869.py` 共 208 项变异。首轮 193 CAUGHT / 10 SURVIVED / 4 ERROR，其中 4 个 ERROR 是变异定义本身的问题（两个 `t1 = 2*_sqrt(A)*alpha` 在 deemph 与 bass 各出现一次、`waveform_shape` 锚点在 call 与 coeffs 各出现一次、census 多行替换破坏缩进），改用显式出现序号与完整块替换修复，并补一项 `ca-shape-coeffs` 变异。
10 个存活项中 **5 个是判题盲区**，分两轮补硬化常量与检查后全部捕获：第一轮 4 个 —— `ie-div-sign`（负采样率下 `Q=0` 给 `-inf`）、`ie-exp-inf`（`band` 的 `Q<0, sr=0` 给 `+inf`）、`ie-shape-msg`（ragged 波形消息）、`ca-clean`（`call` 的非有限系数必须清洗成字符串）。第二轮由 `equiv869.py` 首跑发现 `ie-sqrt-neg` 并非真等价：`band` 取负 Q 且 `noise=True` 时判别式内是**有限负数**，官方 `torch.sqrt` 返回 `nan` 而 Python `math.sqrt` 会抛 `ValueError`，因此它是第 5 个盲区，补 `check_band_noise_negative_discriminant` 后捕获。fixtures 由 117 增到 132 个常量、checks 由 49 增到 53 个。
其余 **5 个是语义真等价**，由 `equiv869.py` 在 **13128 配置 × 9 mode = 每个变异 118152 次、合计 590760 次**完整 `solve()` 输出精确比较（含异常类型与消息、float 逐位）证明零差异：`cf-args-bass` / `cf-args-biquad` 因 ARGS 只进 dict comprehension（字典相等不看顺序）、`ie-err-type` 因 `_err` 仅在 `errors` 里且两类异常恰为 ValueError、`ri-b-re` / `ri-a-re` 因 cos 是偶函数且 `math.cos(-y)` 与 `math.cos(y)` 逐位相同。**不删除任何变异**，最终 208 项剔除 5 个真等价 / 203 项有效全 CAUGHT / 0 SURVIVED / 0 ERROR。

## 6. 工程教训

① torch 张量除零返回 `inf/nan` 而 Python `math` 抛异常，参考解必须用 `_div` / `_sqrt` 等显式模拟；② `-inf` / `nan` 不是合法 Python 字面量，fixtures 必须归一成稳定字符串，否则判题文件无法解析；③ 同一锚点在多个函数出现时，变异脚本必须用显式序号定位，否则整项报 ANCHOR-ERROR；④ 混合符号 inf（`Q=0` 配负采样率）能区分裸 `float('inf')` 与按符号构造的 inf，是补齐 IEEE 层的关键用例。

落地 11 文件、selftest 双模式、双语校验、注册 847 题、短护栏与独立 `--basetemp` 全量护栏、日志回填。改完先不 git commit，等用户确认。
