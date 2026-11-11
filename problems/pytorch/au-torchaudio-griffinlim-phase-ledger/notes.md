# 860 出题笔记：torchaudio Griffin-Lim 相位重建账

## 1. 选题与查重

order 860，合集 `audio` 11→12，slug `au-torchaudio-griffinlim-phase-ledger`。取 `functional.py` 148–353 行四函数 `inverse_spectrogram` / `_get_spec_norms` / `_get_complex_dtype` / `griffinlim`。全库 `griffinlim` / `inverse_spectrogram` / `loudness` / `pitch_shift` / `spectral_centroid` / `mvdr` / `beamform` / `edit_distance` / `frechet` 零命中；`mask_along_axis` 已被 order 693 `asr-specaugment-mask` 覆盖，避让。MVDR 九函数需纯 Python Hermitian 特征分解，风险高列为备选；`loudness` / `pitch_shift` / `_stretch_waveform` 与 857 的 `phase_vocoder` 部分重叠。
与 855 / 857 / 858 / 859 同源于 `functional.py` 但函数集合不重叠；与 344 多相重采样按算法族划界。

## 2. 官方逐字环境

`.workbuddy/scratch/torchaudio860/official860.py` 用 AST 抽四函数 exec，注入 `math/typing/warnings/torch/Tensor/Optional/Union/_CPU`；封装 `comp_norms` / `comp_complex_dtype` / `comp_griffinlim` / `comp_inverse` / `raw_istft` / `raw_stft` / `rand_angles` / `window_l2`。★ 关键工程决策：不剥除 `rand_init=True`，而是在驱动层用 `angles_override` 临时替换 `torch.rand` 为固定 complex128 矩阵，从而把随机初相位变成显式入参，对齐全库 RNG 显式入参化惯例。
回源阶段钉到的两条可行域约束：`torch.istft` 的 OLA 分母要求窗首尾不能同时全零（对称 Hann + `length` 拉长直接报 `window overlap add min`）；`griffinlim` 迭代里 `rebuilt` 的帧数是 `1 + length//hop`，必须与输入 `specgram` 帧数一致。

## 3. 十二段探针

```text
P1  verbatim source / P2  _get_spec_norms / P3  _get_complex_dtype
P4  momentum guard + n_iter=0 / P5  istft OLA, center, length
P6  reflect-pad framed DFT / P7  win_length < n_fft zero padding
P8  Hermitian mirror / DC-Nyquist imaginary
P9  normalized + window L2 / P10 inverse_spectrogram quirks
P11 rand_init / explicit angles / P12 griffinlim end-to-end
PROBE 860 DONE ok=105 fail=0
```

首轮 2 红均为探针期望值写错：`hop=2` 时 16 点补零波形给出 5 帧而不是 7 帧；`n_iter=0` 依然使用 `rand_init` 的初相位（官方先初始化 angles 再进循环），故两种 rand_init 输出不同。两处都改成按官方实测表述，反而固化成两个 quirk。

## 4. 三层对拍

`verify860.py` 对九组参数（`n_fft=8/10/16`、`hop=2/3/4/5/8`、矩形窗 / period  Hann / `win_length=6`、`center` 两种、`normalized` 两种、`length` 多种）逐项比对「官方逐字环境 / 参考解 / 独立 oracle」。oracle 刻意不用 `cmath`：把 Hermitian 逆 DFT 展开成实余弦/正弦对，并以逐输出样本 gather 各帧代替 scatter。首轮 2880 项全绿但有 16 个 case B 的 GL 用例因 reflect 填充长度不足被跳过；把 GL 帧数按 `T = n_fft//2//hop + 3` 提升后 **2960 checks ALL GREEN / 0 SKIP**。

## 5. 判题用例

`test_cases_staged.py` 共 45 个顶层 `check_*(solve)`，数值全部来自 `fixtures860.txt` 的官方逐值快照；`driver860.py` 与 selftest run/submit 均 `45/45 ALL ACCEPTED`。首轮 32 个用例时 7 红，其中 5 个是 `**GEOM` 展开与显式覆盖参数冲突（改成 `_g(**overrides)` 工厂），2 个是 `envelope_min` 期望写成 1.0（矩形窗 `hop=4` 的输出区间 envelope 恒为 2.0）。

## 6. 变异测试

```text
round 1: 134 CAUGHT / 92 SURVIVED / 0 ERROR
fix    : _near() must assert, not return a bool
round 2: 178 CAUGHT / 48 SURVIVED / 0 ERROR
patch  : +11 hardening checks, 4 equivalence rulings
round 3: 220 CAUGHT / 2 SURVIVED / 0 ERROR
patch  : +2 hardening checks (n_fft=16 default, zero phase)
round 4: 222 CAUGHT / 0 SURVIVED / 0 ERROR
226 total / 4 equivalent / 222 active -> ALL MUTANTS CAUGHT
```

★ 首轮 92 个 SURVIVED 是判题辅助函数写错：`_near` 返回布尔值而不是内部 `assert`，导致所有裸调用 `_near(...)` 实际未断言。改回历史约定（递归 `assert`）后立刻收到 48 个。剩余 48 个来自未测的默认值、只校验关键词不校验全文的错误消息、bool / 字符串 / 非数值类型分支、未断言的元数据字段、以及缺失入参分支；追加 11 组硬化用例后收口到 2 个。
最后 2 个：`g-win-default` 要 `n_fft=16` 且省略 `win_length` 才能暴露默认 8 与默认 `n_fft` 的差别；`l-norm-noeps` 需要一次「重建谱幅恰为 0」的迭代，用全零 `angles_init` 令 `spec*angles` 全零、重建谱全零，官方靠 `+1e-16` 避免 `0/0` 并输出全零波形。两条硬化用例都从官方逐字环境取期望值。
4 项真等价：`require_float` 从不带 `low` 下界调用，故其下界分支与消息不可达；`win_length == n_fft` 时通用补零路径与直接返回相同；`buf[i] if i < total else 0.0` 与 `0.0 if i >= total else buf[i]` 字面等价。

## 7. 剥除口径

① 判题域只取二维 `(freq, frames)` 谱，官方的 pack/unpack 批次维不纳入；② `rand_init` 的 RNG 改为显式 `angles_init`，官方 `torch.rand(complex128)` 的值由固定 seed 快照提供；③ 复数矩阵用 `real` / `imag` 两份实矩阵表达，`angles_init` 用 `[re, im]` 对；④ `pad_mode` 记录为「签名存在但从未使用」而不模拟；⑤ 数值统一 float64，`float32` / `float16` 只在 `complex_dtype` 的映射表里出现；⑥ `torch.fft` 用 O(N^2) 复指数 DFT 重写，独立 oracle 用三角展开交叉验证。

## 8. 官方 quirk 清单

与 `description.md` 第 8 节一致：reflect 排除边界、左偏零补、Hermitian 丢虚部、窗包络除法与 NOLA 阈值、length 补零、normalized 方向相反、norms 的 True/字符串等价、pad 与 pad_mode、momentum 变换、tprev 原地与未归一化副本、+1e-16、n_iter=0 仍用初相位、帧数与 length/hop 约束。

## 9. 查重与落地

落地前确认 `max_order=859`、`griffinlim` / `inverse_spectrogram` 全库零命中；`collections.json` 用最小侵入注册，`todos.md` 更新为 838 题 / audio 12 / 下一批从 861 起。11 文件由 `land860.py` 落地，三件套与 `variants/pytorch/` 用 `filecmp` 校验一致。
