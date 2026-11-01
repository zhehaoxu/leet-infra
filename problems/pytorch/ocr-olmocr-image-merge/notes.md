# notes — ocr-olmocr-image-merge (order 692)

## 出处

- 官方源码：allenai/olmocr `olmocr/prompts/anchor.py` 的
  `_merge_image_elements(images, tolerance=0.5)`（GitHub `main`，2026-11
  抓取；配套 `tests/test_anchor.py` 只测整体锚文本，未覆盖合并细节）。
- 调用链：`get_anchor_text(pdf, page, "pdfreport")` → `_pdf_report`（pypdf
  `extract_text` 的 visitor 收集 `TextElement`/`ImageElement`，图片 bbox 由
  CTM 变换单位方框 `[0,0]-[1,1]` 得到）→ `_linearize_pdf_report`，后者
  先 `_merge_image_elements` 再按长度预算线性化。
- 回源取舍：`_linearize_pdf_report` 的超预算路径含 `random.shuffle`
  （无种子，不可复现）、`_cleanup_element_text` 依赖 `fttfy`（判题环境
  无）、`_cap_split_string` 的 head/tail 截断 quirk 有趣但单薄——均不考；
  `_merge_image_elements` 是完全确定性的 union-find 账，立题。

## 算法逐条账

1. 分离距离公式：`h_dist = max(0, max(x0s) - min(x1s))`。两框分离时
   `max(x0s) >= min(x1s)` → 正的 gap；重叠/触碰时为负或 0，被 clamp 到
   ≤ tol。**闭区间** `<= tolerance`（gap 恰等于 tolerance 合并，实测
   锚点 2b/2c 确认 0.5 合并、0.6 不合并）。
2. `max(0, ...)` 截断的等价性：负距离（真实重叠）去 clamp 后仍是负数，
   同样 `<= tol` → 去截断是**数学等价变异**，未收录（notes 里说明）。
   而 min/max 写反（`min(x0s) - max(x1s)`）会把正 gap 变负 → 立即可观察。
3. union 方向 `parent[root_i] = root_j`（i 的根挂到 j 的根）→ 组件最终根
   偏向后出现的元素，但 `groups` 按 `find(i)` in 0..n-1 插入 → **输出
   顺序 = 各组件最小下标升序**（与根是谁无关）。因此 union 方向反转、
   路径压缩开关都是等价变异，未收录。
4. 名字拼接：`indices` 是升序（`groups.setdefault(...).append(i)` 按扫描
   序），`merged_name += f"+{name}"` → **按下标升序拼接**。锚点 4 钉死
   `["z"(0.8,...), "a"(0,0,1,1)]` → `"z+a"`：union 时 j 的根赢了也无妨。
5. bbox 聚合从 `images[indices[0]]` 起逐成员扩矩形——等价于全体成员的
   逐维 min/max。
6. 双重循环 `i<j` 全对扫（O(n²) 邻接判定），不是只扫相邻对——
   `adjacent-pairs-only` 变异靠"隔位重叠"锚点（check_random 与链式锚点）
   抓住。

## oracle 设计

`_oracle` 用邻接表 + 显式栈 BFS 求连通分量（无 union-find），分量按 start
升序发现即天然按 min index 有序，聚合用生成器 min/max。与 solve 300 组
随机对拍（坐标 1 位小数、tolerance ∈ {0, 0.25, 0.5, 1.0}，含大量恰好
贴合/恰好等于容差的临界构型）。

## 变异记录

13 条全抓：tol-exclusive、dist-and-to-or、h-dist-minmax-swap、
v-dist-ignored、adjacent-pairs-only、name-no-plus、name-last-wins、
name-reversed-indices、bbox-first-only、bbox-minmax-x-swap、
tolerance-doubled（恒真连接词）、self-union-only（首轮替换串误伤
`def union(i, j)` 定义行报 SyntaxError，改为限定调用点后重跑）、
groups-key-original-index。等价变异 2 条（去 max(0,...) 截断、union
方向反转）分析后未收录，理由见上。

## 查重

- 全库无 union-find/连通分量类题（`grep union|dsu|merge` 命中的均为
  权重合并/patch merge 语义）。
- 与 `ocr-layout-column-sort`（order 691）同日相邻题：那题是规则状态机
  排序，本题是图连通性聚合，机制无交。
- `mm-patch-merger`/`mm-prumerge-*` 是视觉 token 合并（注意力分数驱动），
  与 bbox 几何容差合并不同。
