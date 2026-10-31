# CornerNet corner pooling 前向账

## 背景与位置

CornerNet（1808.01244）不用 RoI，而是把物体框的左上/右下角点建模为热图极值。为了让角点「看到」框内侧信息，设计了 corner pooling：对左上角点，逐点取其**右侧**与**下方**的最大值再相加。官方实现在 `_cpools`（C++/CUDA 四方向 pool），本题转写其 forward 为纯账本。

## 精确账目

1. **四方向 running max**（对 [H][W] 逐行/列，扫描含自身）：
   - `right_pool`（官方 right_pool.cpp）：向右扫描累积，`out[i][j] = max(x[i][0..j])`——每点取**左侧** max。文件名指扫描方向，语义取左 max，反直觉。
   - `left_pool`：向左扫描累积，`out[i][j] = max(x[i][j..W−1])`——取**右侧** max。
   - `top_pool`：向下扫描累积，`out[i][j] = max(x[i..H−1][j])`——取**下方** max。
   - `bottom_pool`：向上扫描累积，`out[i][j] = max(x[0..i][j])`——取**上方** max。
2. **corner 组合**（官方 models/CornerNet.py：`tl_pool = (TopPool, LeftPool)`、`br_pool = (BottomPool, RightPool)`）：
   - tl = top + left（下方 max + 右侧 max，与论文「max to the right + max to the bottom」一致）
   - br = bottom + right（上方 max + 左侧 max）
   - tr = top + right、bl = bottom + left（官方只实现 tl/br，tr/bl 按对称性）
3. **两支路各自 pool 后逐元素相加**——不是先加再 pool（顺序影响结果）。
4. running max 的结构性：out ≥ x 逐元素成立、out 与 x 等形、单调扫描 O(HW)。

## 约束

1. x 为非空矩形 [H][W]，全部有限浮点；mode ∈ {tl,tr,bl,br}。
2. 单行/单列退化合法（单行时 top/bottom 支路=自身）。
3. 禁改入参；输出浮点列表。

## 数据规模

- H, W ≤ 512（O(HW) 扫描账）。

## 验收锚点（节选）

- `x = [[1,5,2],[3,0,4]]`：tl = `[[8,10,6],[7,4,8]]`（top=`[[3,5,4],[3,0,4]]` + left=`[[5,5,2],[4,4,4]]`）；br = `[[2,10,7],[6,8,8]]`。
- 全 2.0 输入 → 任意 mode 输出全 4.0。
- 全负值输入 running max 仍含自身（`bl` 锚点见测试）。
