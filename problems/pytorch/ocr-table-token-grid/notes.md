# notes — ocr-table-token-grid（order 689）

## 回源

- 出处：`PaddlePaddle/PaddleOCR` `ppstructure/table/tablepyxl/tablepyxl.py` 的 `write_rows`（PP-Structure `TableSystem` 导出 Excel 的官方通道；上游 tablepyxl 项目被引入）。属性解析 `string_to_int`：`s.isdigit() → int(s)` 否则 **0**。
- 官方三段机制：① `cell = ws.cell(row, col)` 后 `while isinstance(cell, MergedCell): col += 1` 的占用跳过；② `colspan/rowspan > 1` 时 `merge_cells(start_row, start_column, end_row, end_column)` 且 anchor 是左上角；③ `col += colspan` 后行末 `col = initial_column`、`row += 1`。
- **quirk 三件**（均已钉题）：属性非数字 → 0 → 不 merge 且 `col += 0`，下一 cell **覆盖同格**（cells 账目出现两个 cell_id 同 (row, col)）；全角数字 `"２"` 满足 isdigit → `int("２") = 2`（Python 语义，官方路径真实行为）；merge 区域非 anchor 格在 openpyxl 里值清空（grid 存 None）。
- **merge 扩展行**：openpyxl `merge_cells` 自动扩展 sheet——rowspan 越过末行时 `num_rows` 按 merge 右端增长（随机对拍抓出初版 bug 后钉入口径与锚点）。
- 与 vision/OCR 系划界：SLANet 结构 token 本身如何生成（模型前向）不出题，本题只考**token 序列 → grid 的展开账**；TEDS 树编辑距离是另一机制（评估侧），不入题。
- 判题环境无 openpyxl/lxml——输入直接结构化为 `[text, colspan_attr, rowspan_attr]`（属性字符串原样），merge/占用语义按 openpyxl 存储语义纯 Python 复刻（题面钉死，注明简化：merge 区域重叠的非良构表不出）。

## 判题面

- solve 入参：rows（[[text, cs_attr, rs_attr], ...]，属性 str|None）。返回 cells/merges/num_rows/num_cols/grid。
- oracle：`test_cases._py` 用 while 循环显式指针 + 字典占用表 + 独立 grid 回填（覆盖优先、再清非 anchor），与 solve 的集合路径独立；200 组随机对拍（属性池含 None/"1"/"2"/"3"/"0"/"abc"/"２"/""）。
- 锚点：表头 colspan=2 → merge [1,1,1,2]、h2 落 (1,3)、grid [0,None,1]；rowspan=2 首列 → 第 2 行 b 从 (2,2) 跳过占用；2×2 矩形 merge 两行跳过；`"abc"`/`"0"` 属性 → 覆盖同格；全角 "２" → colspan 2；末行 rowspan=3 → num_rows 扩到 3；空表/空行；后行拉宽 num_cols。
- 非法输入：rows 非列表/td 非三元组/text 非 str/属性非 str|None → ValueError。

## 变异记录（11 条可观察全抓 + 2 条等价）

occupied-skip-off、col-advance-constant（忽略 colspan 推进）、row-no-reset（行末列不重置）、non-numeric-to-one（非数字属性解析成 1）、merge-end-plus-one、spans-swapped（两属性互换）、merge-condition-and（or→and）、grid-anchor-also-none、cells-zero-based、col-advance-rowspan（col += rowspan）、none-attr-default-zero——全抓。**等价变异**：anchor-occupied（col 单调推进不回退 + 行重置换 row key，merge anchor 永不被重访，加入 occupied 不可观察）；isdigit-to-isdecimal（两者差异字符是 No 类如 `²`，`int()` 均崩溃，本题无异常输出契约）。

## 红线

真图/模型前向/PIL 不出，token 序列结构化入参；merge 区域重叠的非良构表不出（openpyxl 重叠 merge 语义随版本变动，题面不钉）；TEDS/表格评测指标不入题（归 evaluation 侧候选）；`ocr-layoutlm-bbox-index` 已占文档 bbox 位置索引勿混。
