# MinerU 检测框的区间裁除、重叠门限与 span 行聚合账

## 背景

`opendatalab/MinerU`（`master`）的 OCR 后处理里有四段与检测模型本身无关的纯数值/整数处理：bbox 在 y/x 轴上的重叠比例门限、闭区间排序合并与遮罩裁除、span 按行聚合（含竖排判定与特殊类型断行）、以及检测四点框的 angle 隔离与宽高比触发的同行横向合并。本题把四段抽成**无状态判题**：bbox 一律 `[x0, y0, x1, y1]` 四元组整数列表，`poly` 一律 4 个 `[x, y]` 点，span 类型一律字符串；不跑真实检测模型、不加载图像、不依赖 torch/PIL。实现单入口 `solve(mode, **kw)`，共 4 个 mode。

官方 `master` 已重构，旧 `span_block_fix.py` / `para_split.py` 路径不在；本题只取 `mineru/model/ocr/geometry.py` 与 `mineru/backend/analysis/pdf/text/lines.py` 里可确定性复现的纯账。

## 通用约定

所有 mode 返回 dict，字段语义见「模式与规则」；未列出的中间量不判。阈值 / 常量均作显式入参，判题会显式传入，下列默认值仅作口径说明：

```
OVERLAP_DEFAULT = 0.8
LINE_OVERLAP = 0.6
VERTICAL_RATIO = 2
VERTICAL_DOMINANCE = 0.8
LINE_WIDTH_TO_HEIGHT = 4
SPECIAL_TYPES = {"interline_equation", "image", "table"}
```

两点全局口径：**(1) 所有比例门限都是严格 `>`，等于阈值一律不通过；(2) `lines.py::merge_spans_to_line` 调用重叠门限时显式传 `threshold=0.6`，而 `update_det_boxes` 用默认 `0.8`，两处阈值不同，必须分别覆盖。**

## 模式与规则

### 1. overlap_gate：y/x 轴重叠比例门限

输入 `bbox1, bbox2, axis="y", threshold=0.8`。返回 `{"overlap", "min_extent", "ratio", "exceeds", "zero_extent"}`。

```
y: overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2));  e = min(h1, h2)
x: overlap = max(0, min(x1_1, x1_2) - max(x0_1, x0_2));  e = min(w1, w2)
if e <= 0: ratio = None; exceeds = False; zero_extent = True
else:      ratio = overlap / e; exceeds = ratio > threshold; zero_extent = False
```

- `max(0, ...)` 钳位不可省：完全分离时原始差值为负，必须回 0，且 `ratio` 为 `0.0`（不是负数）。
- 退化框（宽或高为 0）不除零，直接 `zero_extent=True`。
- `ratio` 恰等于 `threshold` 时 `exceeds=False`（严格 `>`）。

### 2. interval_ledger：闭区间合并与遮罩 ±1 裁除

输入 `intervals, masks, original`（`original` 可为 `None`）。返回 `{"merged_intervals", "merged_masks", "remaining", "n_segments", "covered_points", "remaining_points"}`。

```
merge_intervals(ivs): sort by start; if not merged or merged[-1][1] < it[0]: append
                      else: merged[-1][1] = max(merged[-1][1], it[1])
remove_intervals([s,e], masks):
    for (ms, me) in merge_intervals(masks):
        if ms > e or me < s: continue
        if cur < ms: remaining.append([cur, ms - 1])
        cur = max(me + 1, cur)
    if cur <= e: remaining.append([cur, e])
```

- 合并条件是**严格 `<`**：相邻闭区间 `[0,10]` 与 `[10,20]` 会被并成 `[0,20]`。
- 裁除用 **±1 离散整数点**语义：遮罩 `[2,4]` 吃掉 2/3/4 三个点，前段收在 `1`，后段从 `5` 起。
- `original=None` 时 `remaining=[]`、`covered_points=0`、`remaining_points=0`；否则 `covered_points = (e-s+1) - sum(每段点数)`。
- `merged_intervals` 与 `merged_masks` 用**输入副本**计算，不判原地副作用。

### 3. line_grouping：span 行聚合

输入 `spans`（每项 `{"bbox", "type"}`）, `threshold=0.6`。返回 `{"vertical", "valid_span_count", "vertical_span_count", "downgraded", "lines", "line_bboxes", "orientation"}`。

```
1) interline_equation -> inline_equation, count downgraded
2) skip bbox if len < 4 or w <= 0 or h <= 0; valid += 1; h/w > 2 => vertical += 1
3) is_vert = valid > 0 and vertical / valid > 0.8
4) vertical:   outer key bbox[2] desc, group by x-axis overlap, inner key bbox[1] asc
   horizontal: outer key bbox[1] asc,  group by y-axis overlap, inner key bbox[0] asc
5) force break if self.type in SPECIAL_TYPES or any member of current line is
6) line_bbox = [min x0, min y0, max x1, max y1] over the line
```

- `h/w > 2` 与 `vertical/valid > 0.8` **都是严格 `>`**；`h/w == 2` 不算竖排，占比恰 0.8 不判竖排。
- 竖排判定的分母是**有效 span 数**（退化框不计入 `valid`），不是总 span 数。
- 特殊类型断行是「自己或当前行内任意成员」的双向或；`interline_equation` 已降级，故不再触发断行。
- 行内排序键：横排为 `bbox[0]`（x0），竖排为 `bbox[1]`（y0）；与外层分组键不同。
- 默认 `threshold=0.6`（不是 0.8）：`lines.py::merge_spans_to_line` 固定传 0.6，不透传外部阈值。

### 4. det_box_merge：angle 隔离与宽高比触发合并

输入 `polys`, `ratio_threshold=4`, `overlap_threshold=0.6`。返回 `{"angle_flags", "horizontal_bboxes", "line_bboxes", "ratio_triggered", "result_bboxes"}`。

```
is_angle(poly):  height = ((p4.y - p1.y) + (p3.y - p2.y)) / 2
                 return not (0.8*height <= (p3.y - p1.y) <= 1.2*height)
points_to_bbox:  [p1.x, p1.y, p2.x, p3.y]
per poly: angle=True -> angle_polys, else -> points_to_bbox into horizontal
horizontal sorted by y0, grouped by y-axis overlap (overlap_threshold)
per line: lb = bounding box of the line
          trig = (lb.w) > (lb.h) * ratio_threshold
          trig -> merge_overlapping_spans(line) (strict < on x adjacency)
          else -> keep each bbox unchanged
append points_to_bbox(p) for every p in angle_polys at the end
```

- `is_angle` 的 height 用**左右两侧高度的平均**，判定是 `not (0.8h <= dy <= 1.2h)`，即 dy 落在带内为非 angle。
- `points_to_bbox` 取 `p1.x / p1.y / p2.x / p3.y`（不是其它点组合）。
- 宽高比判定是**严格 `>`**：`w == 4*h` 不触发。
- `merge_overlapping_spans` 的条件是 `merged[-1][2] < b[0]`（严格 `<`），故 x 边界恰好相接的两个框会被合并。
- angle 框**不参与**行分组与宽高比判定，只在最后原样追加。
- 默认 `ratio_threshold=4`、`overlap_threshold=0.6`。

## 输入与返回示例

```
solve(mode="overlap_gate", bbox1=[0, 0, 10, 10], bbox2=[0, 5, 10, 15],
      axis="y", threshold=0.8)
# {"overlap": 5, "min_extent": 10, "ratio": 0.5,
#  "exceeds": False, "zero_extent": False}

solve(mode="interval_ledger", intervals=[[5, 8], [0, 3], [7, 12]],
      masks=[[2, 4], [6, 9]], original=[0, 12])
# {"merged_intervals": [[0, 3], [5, 12]],
#  "merged_masks": [[2, 4], [6, 9]],
#  "remaining": [[0, 1], [5, 5], [10, 12]], "n_segments": 3,
#  "covered_points": 7, "remaining_points": 6}

solve(mode="det_box_merge",
      polys=[[[0, 0], [100, 0], [100, 10], [0, 10]],
             [[5, 1], [105, 1], [105, 9], [5, 9]]],
      ratio_threshold=4, overlap_threshold=0.6)
# {"angle_flags": [False, False],
#  "horizontal_bboxes": [[0, 0, 100, 10], [5, 1, 105, 9]],
#  "line_bboxes": [[0, 0, 105, 10]], "ratio_triggered": [True],
#  "result_bboxes": [[0, 0, 105, 10]]}
```

## 不进判题域

真实 layout/VLM/OCR 检测前向、PIL/torchvision、numpy 四点数组与 `sorted_boxes` 的整页阅读顺序（已由 `ocr-layout-column-sort`、`ocr-deepseek-tile-ledger` 等题覆盖）、标题分级的 LLM 调用、`_natural_language_spans` 的 LaTeX 定界符切分、表格跨页合并与 markdown 渲染。判题只考上述四段纯数值/整数账。
