# 实现笔记：torchaudio 高斯 Fréchet 距离账

## 1. 选题与查重

863 = `au-torchaudio-frechet-distance-ledger`，order 863 空闲、slug 未占用、audio 合集 14 题。`frechet` / `Fréchet` / `FID` / `eigvals` / `sqrtm` / `Dowson` 在全库 meta、description、notes 零命中（`frechet` 仅在 860 的 notes 中作为候选被提及）。回源纠正了 `todos.md` 的旧说法：本函数是 FID 的高斯距离公式，不是离散 Fréchet / DP，也不是 Levenshtein 编辑距离；后者已有 `ocr-rec-norm-edit-dis` 占位。

## 2. 逐字执行环境

`official863.py` 从 `.workbuddy/scratch/captum855/functional.py`（2511 行）用 `ast.get_source_segment` 抽出 `frechet_distance` 并 exec 到共享 NS，官方函数体未改、未 mock。`_rc` 把嵌套列表直接转成 float64 张量（非数值叶子原样透传，让官方抛自己的 AttributeError）；`ledger()` 按官方逐行表达式重算中间量，`sorted_eigvals()` 消除 LAPACK 根顺序的实现定义。

## 3. 流水线结果

12 段探针 `159/159 ALL GREEN`（源码逐字、五条校验、链式比较真值表、a/b 项、A 与谱、复主值平方根、c 项、装配、退化、census）。三层对拍 `2238 checks / 0 fails / 52 configs ALL GREEN`：第一层官方 LAPACK + torch.sqrt；第二层参考解 Newton 恒等式 + Cardano 闭式 + cmath.sqrt；第三层 oracle Faddeev-LeVerrier 矩阵递推 + Durand-Kerner 迭代 + 手工极坐标平方根，并对无负实谱的 A 额外用 Denman-Beavers 矩阵平方根独立求 `Tr(sqrt(A))`。三条路径无共享算法。判题 `107/107 ALL ACCEPTED`。

## 4. 变异测试与真等价

首轮 137 项 / 116 CAUGHT / 18 SURVIVED / 3 锚点冲突。补 6 条硬化用例（错误消息必须含 `torch.Size([2])` / `[1]` 与 `[3, 3]` / `[2, 2]`、tuple 入参、空 μ 的形状不一致、Σ 内层非数值叶子、实根虚部必须精确 0.0）并修 3 处 occurrence index 后，`127/127 CAUGHT / 0 SURVIVED / 0 ERROR`。10 个真等价写入 `mut863.py::EQUIVALENT` 并附证明：`l-cp-order`（A 与自身幂可交换）、`s-sqrt-neg`（共轭根置换且 Re sqrt 不变）、`v-return-swap`（a/b 对称 + AB/BA 同谱）、`h-shape-descend`（_pack 已保证矩形）、`h-pack-cast`（float() 对数值叶子是 no-op）、`h-pack-width`（ragged 检测只需任一参考行）、`r-n3-w-imag`（w 与 w² 互换，谱排序后相同）、`r-n3-branch`（实三次主值立方根满足 uv=-p/3）、`k-den-skip`（判题域 N=4 对角谱分离，分母不趋零）、`k-ret-old`（break 阈值 1e-14 远小于判题容差）。

## 5. 剥除口径与工程教训

判题域固定 N≤4，μ 与 Σ 全显式入参；不做真实模型前向、不读音频文件、不依赖 scipy / numpy / torch。最大的坑是 `charpoly` 的系数方向：Newton 恒等式自然给出升幂系数，而求根函数需要 monic 降幂，第一次对拍因此全红；改为 `[(-1)^k e_k]` 后 11 组配置误差降到 1e-14 以内。第二个坑是三重根：Durand-Kerner 对 I3 会分裂出约 4e-6 的虚部，所以参考解 N≤3 坚持用闭式、独立 oracle 的 c 改走 Denman-Beavers 矩阵平方根。第三个坑是 `F(X,X)=0` 只在 Σ 谱非负时成立，负定与复谱输入必须单独钉死。

## 6. 官方 quirk 与查重结论

与 `description.md` 第 9-10 节一致：链式比较短路让「方阵 Σ 配错长度 μ」静默通过；官方 `.sqrt().real.sum()` 先开主值平方根再取实部；官方不校验 PSD；`AB` 与 `BA` 同谱使 F 对称。补充：`torch.trace` 对非方阵取短对角，所以非方阵输入是在下游 `eigvals` 而非 trace 处报错；`F_BOOL` 与 `F_INT` 证明 bool / int 叶子被 `torch.tensor(..., dtype=float64)` 静默转换。全库查重确认与 855 / 857-862 的函数集合完全不重叠。

## 7. 注册与护栏

`register863.py` 最小侵入注册 `audio` 14→15、中英 description 各追加 863 口径账长段；复核 `841 registered = 841 disk / max_order=863 / 无 dangling / 无 unregistered`。`todos.md` 更新为 841 题 / audio 15 / 下一批从 864 起。双语全库校验预期 `0 error / 15 warning`（15 条全为历史既有）；短护栏用独立 `--basetemp`，全量护栏后台运行后回填结果。按约定改完先不 git commit，等用户确认。
