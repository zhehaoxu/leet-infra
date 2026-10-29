# repeat_interleave（逐元素重复与 index_select 等价）

## 题目描述

`repeat_interleave` 与 `repeat`/`tile` 是两件不同的事，却是最常被混用的一对：对 `[1,2,3]` 重复 2 次，前者给 `[1,1,2,2,3,3]`（逐元素相邻成块），后者给 `[1,2,3,1,2,3]`（整段平铺）。展开 KV 头的 GQA 复制、把 token 级掩码扩成 token×head、RoPE 的 cos/sin 沿 head_dim 成对复制、把 `cu_seqlens` 差值当重复次数还原变长序列，用的都是它。

$$\text{out}\Big[\sum_{j<i} r_j + k\Big] = x[i], \qquad 0 \le k < r_i$$

即输出第 $\sum_{j<i} r_j + k$ 个位置取自输入沿该轴的第 $i$ 个元素，因此最省事的等价写法是**先构造下标向量、再 `index_select`**。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `repeats` 是非负 int，或**一维整数张量**（逐位置次数，长度必须与沿 `dim` 的元素个数相等）；出现负数、浮点张量、二维张量、长度不符都抛 ValueError。
- `dim=None`：先把 `x` 展平成 1-D 再逐元素重复，返回 1-D 张量，长度 $=\sum_i r_i$；`dim=int` 时只沿该轴重复，其余维原样保留，负数按 `x.dim()` 折算，越界抛 ValueError。
- `repeats` 中出现 0：该位置的元素整个消失（输出长度可为 0，此时返回**空张量但保留轴数与 dtype**）。
- 禁止直接调用 `torch.repeat_interleave`；输出必须是新内存（判题会 `mul_` 输出并断言输入未变）。

## 示例

$X=[[1,2],[3,4]]$（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
X                    = [[1, 2],
                        [3, 4]]

solve(X, 2)          = [1, 1, 2, 2, 3, 3, 4, 4]     # dim=None 先展平
solve(X, 2, dim=0)   = [[1, 2], [1, 2], [3, 4], [3, 4]]
solve(X, [1, 3], 0)  = [[1, 2], [3, 4], [3, 4], [3, 4]]
solve([5, 6, 7], [2, 0, 1]) = [5, 5, 7]
```

第 3 行是逐位置次数不同：第 0 行重复 1 次、第 1 行重复 3 次。第 4 行里 `6` 的重复次数是 0，于是整个消失——输出长度 $\sum r_i = 3$。

## 约束

- 输入 float32，输出 dtype 与形状轴数与输入一致；与 `torch.repeat_interleave` 逐点**精确相等**（`torch.equal`，无浮点容差）
- `repeats` 覆盖 int / 随机长整型张量 / int32 / 全 0 张量；`dim` 覆盖 None、正轴、负轴、中间轴
- 非法入参（负次数、浮点 repeats、二维 repeats、长度不符、越界 dim）逐个断言 ValueError
