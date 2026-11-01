def _string_to_int(s):
    """官方 `string_to_int`：isdigit 才转 int，否则一律 0（注意 Python isdigit 对全角数字也为 True）。"""
    if s.isdigit():
        return int(s)
    return 0


def solve(rows):
    """表格结构 token 展开成 cell 网格账（PP-Structure tablepyxl `write_rows` 口径）。

    输入
    ----
    rows : [[td, ...], ...]，td = [text, colspan_attr, rowspan_attr]
           text 为 str；属性为 str（HTML 原样）或 None（无属性 → 官方 get(attr, "1")）

    记账（题面钉死，与官方 write_rows 一致，起始格 (1,1)，行列均为 1-based）：
      1. 逐行（tr）逐格（td）：先做**占用检查**——若当前格被先前 merge 覆盖
         （openpyxl 的 MergedCell），列指针右移直到可用格；
      2. colspan/rowspan 由属性经 `string_to_int` 解析：`isdigit` 才转 int，
         **否则一律 0**（`"abc"`/`"0"` → 0）；属性缺省（None）按 `"1"`；
      3. 分配 (row, col) 给该 cell；若 rowspan>1 或 colspan>1 触发 merge，
         区域 [row..row+rowspan-1] × [col..col+colspan-1]，anchor 是
         (row, col) 本身，区域内其余格标记为被占；
      4. 列指针 `col += colspan`——colspan 解析为 0 时不推进，下一个 cell
         会**覆盖同一格**（官方 quirk）；
      5. 行结束列指针重置为 1，行指针 +1；
      6. grid：每格存最后分配到该格的 cell_id；merge 区域的非 anchor 格为
         None（openpyxl MergedCell 值为空的存储语义）；从未分配的格也为 None。

    返回
    ----
    dict：
      cells     list[[cell_id, row, col, text]]  分配序账目
      merges    list[[r1, c1, r2, c2]]           触发 merge 的区域（闭区间，1-based）
      num_rows  int                              网格行数 = max(len(rows), merge 右端行)
      num_cols  int                              全表最大占用列（含 merge 右端）
      grid      list[list[int|None]]             num_rows × num_cols 的归属账
    """
    raise NotImplementedError
