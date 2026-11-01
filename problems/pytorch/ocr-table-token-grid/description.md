# 表格结构 token → cell 网格分配账

## 题意

表格识别模型（如 PP-Structure SLANet）输出的是**结构 token 序列**：`<tr>`/`<td>`/`colspan`/`rowspan` 组成的 HTML。要把它变成可用的电子表格，需要把每个 cell 分配到网格坐标上——这正是 PP-Structure 导出 Excel 的官方通道 `tablepyxl.write_rows` 做的事。本题转写这套账本：给定逐行的 cell 列表（文本 + colspan/rowspan 属性原样字符串），输出每个 cell 的 (row, col) 分配、merge 区域与最终归属网格。

## 求解

口径题面钉死（与官方 `write_rows` 一致，行列均 1-based，起始格 (1,1)）：

1. 逐行逐格：先做**占用检查**——若当前格被先前 merge 覆盖（openpyxl 的 `MergedCell`），列指针右移直到可用格；
2. colspan/rowspan 属性经官方 `string_to_int` 解析：`isdigit` 才转 int，**否则一律 0**（`"abc"`/`"0"`/`""` → 0；属性缺省 None 按官方 `get(attr, "1")` 为 1）。注意 Python `isdigit` 对全角数字也为 True，`int("２") = 2`；
3. 分配 (row, col)；若 rowspan>1 或 colspan>1 触发 merge，区域 `[row..row+rowspan-1] × [col..col+colspan-1]`（闭区间），anchor 是 (row, col) 本身，区域内其余格标记被占；
4. 列指针 `col += colspan`——解析为 0 时**不推进**，下一个 cell 会覆盖同一格（官方 quirk）；
5. 行结束列指针重置 1，行指针 +1；
6. `grid`：每格存最后分配到该格的 cell_id；merge 区域非 anchor 格为 None（openpyxl `MergedCell` 值为空的存储语义）；`num_rows` 按 merge 右端扩展——openpyxl `merge_cells` 会自动扩展 sheet，`rowspan` 越过末行时行数随之增长。

返回 `{"cells": [[cell_id, row, col, text], ...], "merges": [[r1,c1,r2,c2], ...], "num_rows", "num_cols", "grid"}`。

## 数据规模

- 0 ≤ 行数 ≤ 100，每行 0 ≤ cell 数 ≤ 50，属性串长度 ≤ 8。
- rows 非列表、cell 非三元组、text 非 str、属性既非 str 又非 None → `ValueError`。
