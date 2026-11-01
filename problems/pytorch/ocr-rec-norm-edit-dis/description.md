# 识别指标：逐行归一化编辑距离账

## 题意

OCR 识别模型评测时，每行样本得到一对 `(pred, target)` 字符串。PaddleOCR `RecMetric` 在逐行算归一化编辑距离（NED）再汇总出 `acc` 与 `norm_edit_dis` 两个指标。本题转写这套账本：给定行对列表与两个行级预处理开关，输出每行的编辑距离、归一化编辑距离、是否严格相等，以及汇总指标。

## 求解

口径题面钉死（与官方一致）：

1. 行级预处理顺序：`ignore_space=True`（默认）先双方 `replace(" ", "")`；`is_filter=True` 再归一化——`filter` 只保留 `string.digits + string.ascii_letters` 中的字符，随后 `lower()`；
2. `dist` 为经典 Levenshtein 编辑距离（插入/删除/替换权重均为 (1,1,1)，**不含** Damerau 相邻换位：`"ab"→"ba"` 是 2 步）；
3. `ned_i = dist / max(len(pred), len(target))`——分母是**较长串**的长度，不是 `len(target)`；两串皆空时分母为 0，`ned_i = 0`；
4. `exact = 预处理后的 pred == target`（严格相等）；
5. 汇总：`acc = exact 行数 / (n + eps)`，`norm_edit_dis = 1 - Σned_i / (n + eps)`，`eps = 1e-5`。空批（n=0）照公式直接计算：`acc = 0.0`、`norm_edit_dis = 1.0`。

返回 `{"rows": [{"pred", "target", "dist", "ned", "exact"}, ...], "acc", "norm_edit_dis"}`，`rows` 内的 `pred`/`target` 为预处理后的文本。

## 数据规模

- 0 ≤ n ≤ 10^3 行，每行长度 ≤ 500，字符串任意 Unicode。
- `pairs` 非列表、元素非二元组/元素含非字符串、开关非 bool → `ValueError`。
