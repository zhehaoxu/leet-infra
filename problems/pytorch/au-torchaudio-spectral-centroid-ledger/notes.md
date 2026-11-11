# 实现笔记：868 `spectral_centroid` 谱质心账

## 1. 官方环境

`spectral_centroid` 从 `functional.py` 1257-1299 行用 AST 抽取后 `exec` 进共享 NS，`torch.set_default_dtype(torch.float64)`。无 Proxy、无 RNG、无音频 IO、无 warning；唯一替身是 `spectrogram` spy，返回显式幅度谱并记录 12 个 kwargs。`comp` 实现八路，与参考解一一对应。

## 2. 参考解

纯 Python 只用 `math`。对称 `linspace`（后半段由 `end` 反向算）逐位对齐 torch；`_shape_of` 复刻 ragged 检测；`_weighted` 复刻 `(F,1)` 广播；`_sum_freq_dim` 复刻 `sum(dim=-2)`；`_div` 把 `0/0` 归一为字符串 `"nan"`、非零除零给 `±inf`；输出形状 `shape[:-2] + shape[-1:]`。八 mode 与官方 `comp` 一一对应。

## 3. 独立 oracle

`verify868.py` 第三层用 `Fraction` 精确有理数独立重算频率栅 / 加权和 / 分子 / 分母，并用 99 条代数性质（单频脉冲质心、区间有界性、列缩放不变性）校验；84 组配置共 43488 项比较零差异，与参考解不共享浮点公式。

## 4. 常量与判题

33 个官方常量 + 50 个 `check_*`：频率栅 13 组、质心 8 组、ledger 3 组、spy kwargs 2 组、零帧 / shapes / errors / census，外加三条硬化（census 新 dict、±inf 分支、shapes 异常分支）。

## 5. 变异与等价

155 项变异剔除 4 个真等价后 151 全 CAUGHT：4 个存活真等价由 `equiv868.py` 在 188 配置 × 9 mode = 6768 次完整输出比较证明；另 4 个存活项属于判题盲区，补硬化用例后捕获。5 个锚点错误用显式出现序号修正。

## 6. 查重与划界

`spectral_centroid` 仅 2 次命中且均为 859 / 860 notes 的排除说明，未覆盖；855 判 `spectrogram` 本体与后处理链，868 只把 spectrogram 作显式入参 spy，判 kwargs 与质心语义。`n_fft<=1` 触发 torch 单 bin 广播 / 空 linspace 实现细节，明确排除在输入域外（同 866 对 `n_freqs=0` 的处理）。

## 7. 交付与约定

落地 11 文件、selftest 双模式、双语校验、注册 846 题、短护栏与独立 `--basetemp` 全量护栏、日志回填。改完先不 git commit，等用户确认。
