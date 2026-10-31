# 出题笔记：vision-cornernet-pooling

## 回源

- `princeton-vl/CornerNet` `models/py_utils/_cpools/src/{right,left,top,bottom}_pool.cpp` forward 循环逐行核对：right_pool `output[0]=input[0]`、`output[i+1]=max_out(input[i+1], output[i])`；left_pool 反向从 W−1 起。
- 组合类在 `models/CornerNet.py`：`class tl_pool(pool): super().__init__(dim, TopPool, LeftPool)`、`br_pool(dim, BottomPool, RightPool)`——组合顺序是位置参数，两个支路各自 pool 后 `pool1 + pool2` 逐元素相加。
- 官方只有 tl/br 两类（网络只有两张角热图）；tr/bl 组合按对称性补充并在 description 声明。

## 关键口径判断

- **文件名与语义反直觉**：right_pool.cpp 是"向右扫描"（out[j] = max(x[0..j])），每点取的是**左侧** max。直觉以为 right_pool 取右侧，变异 `right-suffix`（把累积方向改反）专门钉这个。
- **pool 后相加 ≠ 相加后 pool**：max 不可分配，`(a+b)` 的 running max 不等于两条 running max 之和。变异 `add-to-max`（相加改成取 max）也钉住逐元素相加语义。
- 手算锚点两次踩坑：br 锚点把 1+1 算成 4、单列 br 忘记加 right=自身。solution 与暴力 oracle 对拍一致后再修锚点——教训：逐元素加法手算必须逐格算，不能跳格。
- running max 含自身（含端点），`no-self`/`top-no-self` 变异钉端点初始化。

## 边界划定

- 不做 C++/CUDA、不做 backward（scatter_add 账可作后续候选）；不做 heatmap 解码（与 `vision-heatmap-decode` 查重）；不做 CenterNet 的双池化融合层（结构不同）。
- 与 `vision-ohem-selection`(676) 同批入 vision-detection 合集。

## 测试设计

- oracle `_py_pool` 暴力逐点 O(H·W·n) 重算四个方向，与 O(HW) 扫描解对拍。
- 8 个 check：手算锚点（tl/br 全格）、单行/单列退化、单调行、全等值、全负值、随机 200 组、非法输入 6 档。
- 变异 9/9：四方向各 1 个方向反转、两个端点初始化清零、tl/br 组合支路互换、加法改 max。`pool-then-add-reversed`（支路顺序互换）是等价变异（加法交换律），已剔除记档。
