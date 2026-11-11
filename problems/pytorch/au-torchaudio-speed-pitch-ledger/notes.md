# 864 出题笔记：torchaudio 变速/变调编排账

## 1. 立项与查重

order 864，合集 `audio` 15→16，slug `au-torchaudio-speed-pitch-ledger`。取 `functional.py` 的 `speed`(2385-2423) / `pitch_shift`(1596-1641) / `_stretch_waveform`(1644-1693) / `_fix_waveform_shape`(1696-1719)。全库 `pitch_shift` / `_stretch_waveform` / `_fix_waveform_shape` / `speed` / `bins_per_octave` / `phase_advance` / `len_stretch` 在已立项题的 meta、description、notes 中零命中，只在 860 的 notes 作为候选被提及。与 857 `phase_vocoder`、859 sinc kernel、860 `torch.istft` / Griffin-Lim、862 MVDR、863 FID 的函数集合完全不重叠。

## 2. 官方源码与逐字环境

`official864.py` 用 `ast.get_source_segment` 抽取四个顶层定义并 `exec` 进共享 NS，官方函数体一字未改；`torch` 与 `math` 是转发代理（`stft` / `istft` / `gcd` 额外记录实参），`phase_vocoder` / `resample` 是记录型包装并返回显式入参张量。全局固定 `float64`，保证 `linspace` 与纯 Python 参考解可比。`ledger()` 从调用记录里取 rate / hop / win / window_len / freq_bins / pack_shape / len_stretch / resample rates / gcd 原始值 / out_lengths，是探针与 fixtures 的唯一权威来源。

## 3. 机制账

`rate = 2**(-n_steps/bins)` 与 `int(sample_rate/rate)` 截断；`hop=n_fft//4` / `win=n_fft` / hann 窗默认；`phase_advance = linspace(0, pi*hop, spec_f.shape[-2])` 按**频点数**取长度；`len_stretch = int(round(ori_len/rate))` 半偶舍入；`reshape(-1, ori_len)` 打包；`_fix_waveform_shape` 裁剪/尾部补零 + `view(shape[:-1]+[ori_len])`；`speed` 的两次 `int()` 截断、`math.gcd` 约简后交给 resample、`ceil(lengths*tgt/src).to(dtype)`。

## 4. 三层验证与变异

12 段探针 `427/427 ALL GREEN`（P1 源码逐字比对、P2 rate、P3 int 截断、P4 默认值、P5 linspace、P6 pack、P7 banker's round、P8 stft/istft kwargs、P9 trim/pad/exact、P10 gcd、P11 ceil+dtype、P12 端到端 census）。三层对拍 `13326 checks / 52 configs / 0 fails`：官方走真实逐字环境，参考解走纯 Python 编排，独立 oracle 走 `exp+log` / `n_fft>>2` / `Fraction` 步长 / `Decimal ROUND_HALF_EVEN` / 手写欧几里得 / `Decimal ROUND_CEILING` / 索引驱动分块 reshape，三条路径无共享算法。判题 driver `152/152 ALL ACCEPTED`。

首轮 160 项 / 135 CAUGHT / 25 SURVIVED / 0 ERROR。补 13 条硬化用例（unknown-mode 消息的 MODES 顺序、空 waveform、默认 dtype 必须是 float、两行不同数据的 reshape、1D pad、单频点 linspace、3D spec_f、3D 默认注入的 `_nest`、`fix_shape` 自定义目标长度、`pitch_shift` 不受 `fix_*` 影响、stretched 值不参与输出）后降到 1 个存活项，再把 `d-nest-reverse` 按「同一 row 复制到所有前导维 + `_fix` 展平重排」的证明归档。最终 **160 项剔除 13 个文档化真等价 / 147 全 CAUGHT / 0 SURVIVED / 0 ERROR**。真等价清单：`v-shape-descend`、`v-pack-conv`、`v-pack-no-conv`、`v-neednum-cast`、`v-needint-cast`、`v-torchsize-join`、`n-linspace-zero`、`n-fix-trim-gt`、`n-fix-pad-lt`、`d-stretched`、`d-shifted-len`、`d-nest-reverse`、`l-lengths-reshape`，证明逐条写在 `mut864.py::EQUIVALENT` 注释里。

## 5. 剥除口径与工程教训

判题域不含 STFT/ISTFT 数值、`phase_vocoder` 相位累加、sinc kernel、窗函数数值、音频 IO 与模型前向；波形/谱/窗全部显式入参，纯 Python、无 numpy/torch/scipy、无全局 RNG。最大的坑是**默认注入波形每行都相同**：这会让「reshape 时忽略 idx」「前导维顺序颠倒」这类变异全部存活，必须专门补两行不同数据的用例。第二个坑是 `speed` 的 gcd 原始值只能从 `math.gcd` 的调用实参反推（官方把约简后的频率传给 resample），因此代理对象必须记录 `gcd` 而不是只记录 `resample`。第三个坑是 `spec_f` 的 `-2` 维是频点数：用帧数或 `n_fft//2+1` 硬编码都会在注入不同行数时暴露。

## 6. 官方 quirk 与查重结论

与 `description.md` 第 5 节一致：`int(sample_rate/rate)` 在 20158.738 上给 20158；`linspace` 长度取频点数；`round` 半偶舍入让 4.5→4、7.5→8；`0.7*44100=30869.999...` 被截断成 30869 使 gcd 退化为 1；resample 收到的是 gcd 约简后的频率；`_fix_waveform_shape` 先统一最后一维再 view 回填前导维；`stretched` 在编排层只被校验、不被消费。补充：`n_steps` 经 `float()` 转换允许分数步数，`bins_per_octave=0` 抛 `ZeroDivisionError`；显式 `window` 长度不被校验。全库查重确认与 855 / 857-863 的函数集合完全不重叠。

## 7. 注册与护栏

`register864.py` 最小侵入注册 `audio` 15→16、中英 description 各追加 864 口径账长段；复核 `842 registered = 842 disk / max_order=864 / 无 dangling / 无 unregistered`。`todos.md` 更新为 842 题 / audio 16 / 下一批从 865 起。双语全库校验预期 `0 error / 15 warning`（15 条全为历史既有）；短护栏用独立 `--basetemp`，全量护栏后台运行后回填结果。按约定改完先不 git commit，等用户确认。
