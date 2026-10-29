# 解题说明 · RoAD 的分组二维旋转

## 出处

- `huggingface/peft` 的 `src/peft/tuners/road/layer.py`（RoAD 适配器）：`_prepare_cols` 按 variant 生成 `first_col` / `second_col`，`_get_delta_weight` 用「对角 + 交换半组再取负」拼出旋转阵，`_apply_road` 用 `x * first_col + rotate_half(x) * second_col` 的逐元素形式做推理。本课逐行对齐这份实现。
- RoAD 论文编号未逐一回源，公式以源码为准（notes 只承担口径说明，不宣称与论文逐字一致）。

## 核心与易错点

**配对是 group 内前半与后半，不是相邻元素。** `group_size=4` 时第 0 位配第 2 位、第 1 位配第 3 位；这是为了在张量并行切分后仍能让每对留在同一 rank 内。写成交错配对，`group_size` 较大时结果完全错位。

**`road_1` 的参数要 `repeat_interleave(2)`，不是 `repeat`。** `[θ0, θ1]` 需要变成 `[θ0, θ0, θ1, θ1]`（前半与后半各用一次），`repeat(2)` 会给出 `[θ0, θ1, θ0, θ1]`，角度与配对的对应关系就反了。

**第二列的符号位置容易反。** 分块写法是 `R[a, b] = -second_col[a]`、`R[b, a] = second_col[b]`，两个符号用的都是各自下标的 `second_col`；写成一个用 `a`、一个用 `b` 在 `road_1`（两列同值）里看不出差别，换到 `road_2` 立刻错。

**`road_2` 的 `alpha` 不同就不再是旋转。** 只有每个位置的缩放相同且等于 1 时，分块才是 `α[[c,−s],[s,c]]` 形式的正交旋转；位置间缩放不同时行列长度被改写，`RᵀR = I` 不成立——这是本题特意保留的检查。

**`road_4` 的两个角来自不同段。** `reshape(-1, 2, group_size)` 后 `[:, 0, :]` 供 `cos`、`[:, 1, :]` 供 `sin`；把两段都用同一个角会退化成 `road_2` 的一个特例。

**逐元素公式必须与矩阵一致。** 源码用逐元素形式是为了避免物化 `out×out` 矩阵；判题同时验证两条路径，任何一侧写错都会在 `y == x @ Rᵀ` 上暴露。

**前导维度。** 源码假设输入是 2 维；本题允许任意前导维度，实现时先把前导维展平、算完再 reshape 回去，`rotate_half` 只在最后一维上做交换。

## 与其他题的分工

- `peft-oft-cayley`：OFT 是分块对角正交阵**左乘权重**；RoAD 是作用在**输出**上的分组二维旋转，参数只有角度与缩放。
- `peft-hra-householder-reflection`：HRA 用反射乘积右乘权重；本题是逐对的 2×2 旋转。
- `diff-rf-timestep-shift` 等 diff 侧题目里的 RoPE 式旋转只作用于位置编码；本题在适配器里复用同一套「交换半组 + 取负」的写法。
