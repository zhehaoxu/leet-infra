# olmOCR 图片元素 union-find 容差合并账
#
# allenai/olmocr `olmocr/prompts/anchor.py` 在给 VLM 生成 PDF 页锚文本前，
# 用 `_merge_image_elements` 把相邻（含小间隙）的图片 XObject 合并成一个
# 元素：PDF 里同一张图常被切成多块放置，渲染器逐块记录后需按 bbox 容差
# 并回整图。纯账本：bbox 与名字全入参，不读 PDF。
#
# 实现 solve(images, tolerance=0.5)：
#
#   输入
#     images     list[list]  每项 [name, x0, y0, x1, y1]（bbox 为 PDF 坐标）
#     tolerance  float       合并容差，官方默认 0.5
#
#   输出 list[dict]：每个连通分量一项
#     {"name": 合并名, "bbox": [x0, y0, x1, y1]}
#     输出顺序 = 各分量最小原始下标的升序。
#
# 规则（照抄官方实现）：
#   1. 两框的分离距离：
#        h_dist = max(0, max(x0s) - min(x1s))
#        v_dist = max(0, max(y0s) - min(y1s))
#      两者均 <= tolerance 时判为"重叠"（触碰 dist=0 与恰好 gap=tolerance
#      都算重叠——**闭区间**）。
#   2. 按原始下标做双重循环 i < j，重叠即 union(i, j)（union-find，路径
#      压缩无所谓，只影响根是谁，不影响分量成员）。合并是**传递闭包**：
#      A~B、B~C 但 A、C 自身不相邻 → 三者同分量。
#   3. 分量内按下标升序聚合：
#      bbox 取全体成员的外包矩形（逐维 min/max）；
#      name 取 members[0] 的名字，其余成员名按**下标升序**用 "+" 依次
#      追加（与 union 方向无关——下标小的名字恒在前）。
#   4. 空输入返回 []；单元素返回其自身。
#
# 示例：solve([["a",0,0,1,1],["b",1.5,0,2,1]], tolerance=0.5)
#   h_dist = max(0, 1.5-1) = 0.5 <= 0.5 → 合并
#   → [{"name": "a+b", "bbox": [0, 0, 2, 1]}]
