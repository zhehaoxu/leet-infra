# notes — ocr-layout-column-sort (order 691)

## 出处

- 官方源码：PaddlePaddle/PaddleOCR `ppstructure/recovery/recovery_to_doc.py`
  的 `sorted_layout_boxes(res, w)`（GitHub `main` 分支，2026-11 抓取）。
- 调用场景：PP-Structure 版面恢复导出 docx 前，把 `res`（layout 检测结果，
  每项含 `bbox`/`type`/`res`）按阅读顺序重排；`w` 为原图宽度。docx 导出侧
  再按 `region["layout"]`（"single"/"double"）切换 Word 分节的双栏格式
  （`convert_info_docx` 里的 `flag` 状态机），本题只考排序函数本身。
- 回源探索记录：todos 里原本写"Surya 系或 PP-Structure XY-cut"。实测
  surya 新旧两版阅读顺序全是模型解码（v0.4.15 是 encoder-decoder AR
  逐位输出 rank，master 是 box token cross-attend + 约束贪心 AR），无规则
  账可考，仅 `canonical_order`（y-band 光栅键 `int(y0 // (1000/24))` +
  `(x0, y1, x2)` 次键）可作独立小题，信息量不足——故弃 surya，改立
  PP-Structure 这条规则式实现。

## 算法逐条账

1. `num_boxes == 1`：直接返回原列表（dict 上打 `layout="single"`），**不
   排序**——对单元素无差别，但输出契约里 layout 必为 single。
2. 排序键 `(bbox[1], bbox[0])` 即 `(y0, x0)`，Python `sorted` 稳定 → 键
   相同保持原始顺序（H2 锚点：同 y0 同 x0 的两框按输入顺序输出）。
3. 左桶条件 `x0 < w/4 and x1 < 3w/4`；右桶条件 `x0 > w/4 and x1 > w/2`。
   严格不等号：
   - `x0 == w/4`：两条件均不成立（不 `<` 也不 `>`）→ single（D 锚点）。
   - `x1 == 3w/4`：左桶条件失败；若 `x0 > w/4` 不成立同样 single（E 锚点
     的单框版）。
   - 覆盖缺口：`x0 < w/4` 且 `x1 > 3w/4` 的全宽块两条件都不满足 → single，
     这正是"全宽块作分隔点"的设计意图。
4. 末框三分支 + 丢弃：
   - 横跨分支条件 `y0 > prev.y1 and x0 < w/2 and x1 > w/2`——注意
     `prev` 是**扫描序**前一框（`_boxes[i-1]`），不是原序前一框；
     `>` 严格：相邻框 `y0 == prev.y1` 时不进横跨分支（last-gap 锚点）。
   - 横跨分支 flush **先于** append；列分支 **先** append **后** flush →
     末框在列组内部还是组外，输出顺序不同（I / J 锚点对照）。
   - 丢弃分支条件 `x1 <= w/2 and x0 >= w/2`，与 `x0 <= x1` 联立只可能
     `x0 == x1 == w/2`（退化零宽框）。**丢弃时不 flush 桶**——前面攒的
     左右桶一并消失（F 锚点输出 `[]`）。这是官方最隐蔽的行为。
5. 循环出口分析：`i` 每次 +1，`num_boxes >= 2` 时必然命中
   `i == num_boxes - 1` 的末框分支并 break，桶在该分支内必被 flush 或
   清空；`num_boxes == 0` 在 `i >= num_boxes` 退出、桶为空。因此函数
   末尾的 `if res_left / if res_right` 残余 flush 是**不可达死代码**。
   变异时删掉它属于等价变异，未收录。

## oracle 设计

test_cases 的 `_oracle` 用独立结构重写：手写插入稳定排序（不用
`sorted`）+ 显式 `queue.pop(0)` 扫描 + `pending_left/right` 与
`(idx, layout)` 元组流。与 solve 做 300 组随机对拍（w ∈ {400,800,1000,1200}，
坐标整数，含退化零宽框可生成的构型）。随机坐标下 `y0 == prev.y1 且横跨
中线` 几乎不出现（变异 `last-gap-inclusive` 首轮幸存的教训），靠定向
锚点 `check_last_gap_touching_not_spanning_branch` 钉死。

## 变异记录

13 条全抓：sort-key-swapped、left-cond-half、left-cond-inclusive、
right-cond-3q、cols-swapped、last-gap-inclusive（靠新增相邻框锚点抓）、
span-cond-moved、flush-after-append、last-right-then-left、
drop-becomes-single、single-not-flushed、single-box-sorted-out、
left-right-flip-last。无等价变异幸存（死代码 flush 未做变异）。

## 查重

- 与 `ocr-layoutlm-bbox-index`（bbox 归一化/2D 位置索引）不同点：那题是
  坐标→索引映射，本题是排序状态机。
- 与 347 task DAG 划界：本题无图结构，规则纯几何。
- `w/4`、`3w/4`、`w/2` 是 float 除法，测试全用 4 的倍数宽保证锚点无浮点
  争议。
