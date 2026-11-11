# 实现笔记：torchaudio MVDR 波束形成账

## 1. 选题与查重

order 862 空闲、slug `au-torchaudio-mvdr-beamform-ledger` 未占用、全库 839 题 / max_order=861 / order 唯一。`mvdr` / `souden` / `rtf_power` / `beamforming` / `diagonal_loading` / `Tikhonov` / `_tik_reg` 在全部 meta.json 与 description/notes 均零命中；`psd` 的 meta 命中只有 `rl-sweet-rl-turn-critic` / `rl-opsd-turn-credit`（OPSD 缩写，与功率谱无关）。选它的理由：functional.py 2511 行里 855/857/858/859/860/861 已吃掉频谱后处理、波形算子、NCCF、sinc 重采样、Griffin-Lim、响度门控，MVDR 族（1799-2198 行九个函数）是最后一块未覆盖的大机制簇，复数外积、只改实部的 Tikhonov、solve + 复数迹归一、幂迭代编排、共轭 einsum 都有确定性账。

## 2. 逐字执行环境

`official862.py` 用 `ast.get_source_segment` 从 `captum855/functional.py` 抽出九个顶层定义（含 `rtf_evd`），`exec` 进一个共享命名空间并绑定 `math` / `warnings` / `typing` / `torch` / `Tensor` / `Optional` / `Union`，官方函数体未改。`C()` / `cx()` / `R()` / `py()` 负责 `[re, im]` 与 complex128 / float64 张量的互转；`comp_*` 驱动在返回官方结果的同时用独立转录的 ledger 做 torch.allclose 自校验。`rtf_evd` 只作探针：`.venv-eval` 的 torch 2.14.1 复数 `linalg.solve` / `linalg.eigh` 可用，但特征向量符号不可移植。

## 3. 流水线结果

12 段探针 `PROBE 862 DONE ok=241 fail=0 ALL GREEN`；三层对拍 `VERIFY 862 DONE checks=134334 fails=0 configs=18 ALL GREEN`——第三层 oracle 刻意换算法：复数用 `(re, im)` 浮点对与手写乘除，线性方程组用 2x2 / 3x3 伴随矩阵闭式求逆而非 LU 消去，逐项与官方逐字结果和纯 Python 参考解对齐。`fixtures862.py` 把官方数值固化成 `fixtures862.txt` 常量，判题用例只引用这些手验常量。判题 driver `73/73 ALL ACCEPTED`。

## 4. 变异测试与真等价

四轮迭代：首轮 94 CAUGHT / 34 SURVIVED / 2 ERROR（2 个 ERROR 是唯一锚点误加 occurrence idx 导致文件损坏）；补硬化 fixtures 与 24 条判题用例后第二轮 123 / 7；第三轮 126 / 4；最后修 `check_unknown_mode`（要求错误消息列出 MODES 全表，含 `rtf_power`）并新增 `check_reference_channel_bad_tensor_leaf`（钉死 `pack` 自身的 `reference_channel must be rectangular`）得 128 / 2。最终 `原始 130 项变异，剔除 2 个真等价，有效 128 项：128 CAUGHT，0 SURVIVED，0 ERROR / ALL MUTANTS CAUGHT`。
两个真等价均有证明：`h-unpack-bool`——`pack` 用 `_is_num` 拒绝 bool 叶子，所有 `unpack` 实参都来自 pack 输出（complex / float）或本地构造的数值账，bool 分支在判题域不可达；`l-elim-start`——`_solve` 在消去循环前已 `M[r][col] = 0j`，把循环起点从 `col+1` 改成 `col` 只会往严格下三角写一个非零值，而后续主元搜索只读 `col' > col` 的列、回代只读 `j >= i` 与 RHS 列，严格下三角永不被读，X 逐位相同。

## 5. 剥除口径与工程教训

批次维 `...` 全部剥掉，判题域固定 `(F, C, C)` / `(F, C)` / `(C, F, T)`；复数用 `[re, im]` 对，`bool` 不算数值；`torch.linalg.solve` 用复数高斯消元 + 部分主元复刻，奇异矩阵显式报错而非静默除零。`rtf_evd` 整段排除、改为显式 RTF 入参，是本族最大坑：LAPACK 特征向量符号/相位不确定会让跨实现对拍失效。另需注意 `psd` 与 `_tik_reg` 的 docstring 默认值与签名不一致，一律按签名；`rtf` 的分母 `.real` 与 `souden` 的复数迹是两个最容易互相串味的口径。

## 6. 官方 quirk 与查重结论

与 `description.md` 第 10 节一致：两处 docstring 默认值写错、`withou` / `martrix` 拼写、复数 dtype 检查退化为叶子校验、`v[..., -1]` 选取但 eigh 不可判。补充：官方 `psd` 的 einsum 右操作数取共轭、`sum_dim=-3`；`_tik_reg` 的 epsilon 先乘 reg 再加 eps；`rtf_power` 的 `n_iter - 2` 循环次数与 `psd_s` 尾乘；`apply_beamforming` 的 `w.conj()`。全库查重确认与 855 / 857-861 的函数集合完全不重叠。

## 7. 注册与护栏

`register862.py` 最小侵入注册 `audio` 13→14、中英 description 各追加 862 口径账长段；复核 `840 registered = 840 disk / max_order=862 / 无 dangling / 无 unregistered`。`todos.md` 更新为 840 题 / audio 14 / 下一批从 863 起。双语全库校验预期 `0 error / 15 warning`（15 条全为历史既有）；短护栏用独立 `--basetemp`，全量护栏后台运行后回填结果。按约定改完先不 git commit，等用户确认。
