# FP4 E2M1 网格量化与两级块缩放

FP4 是 Blackwell 一代的原生推理格式：`E2M1` 只有 1 位符号、2 位指数、1 位尾数，
整个正半轴只有 8 个码点。它没有 `inf`、没有 `NaN`、也没有第二个非零次正规——
这些「缺失」正是量化账里最容易写错的地方。

## 接口

```python
def solve(values, block_size, block_scales, global_scale=1.0) -> dict
```

`values` 是 1 维 `float64` 张量；`block_size` 是每块的元素数（MXFP4 用 32，NVFP4 用 16）；
`block_scales` 是逐块的标量列表，长度必须等于 `ceil(n / block_size)`；`global_scale` 是
NVFP4 的第二级（每张量）缩放。返回：

| 字段 | 含义 |
| --- | --- |
| `codes` | 每个元素 4 位码（bit3 符号 / bit2..1 指数 / bit0 尾数） |
| `dequant` | 反量化值（保留 `-0.0` 的符号） |
| `eff_scale` | 逐元素的有效缩放 `global_scale * block_scales[i // block_size]` |
| `sat_mask` / `tie_mask` | 该元素是否饱和 / 是否恰好落在两个码点的中点 |
| `block_sat` / `block_ties` | 逐块的饱和数与平局数 |
| `grid` | 正半轴 8 个码点 |

## 输入输出

```
输入：values=[0.25, 0.75, 1.25, 2.5, 7.0], block_size=5, block_scales=[1.0]
输出：codes=[0, 2, 2, 4, 7], dequant=[0.0, 1.0, 1.0, 2.0, 6.0],
      tie_mask=[True, True, True, True, False], sat_mask=[False, False, False, False, True]
```

`0.25` 是 `0` 与 `0.5` 的中点，平局取尾数位 0 的 `0`；`7.0` 超出最大码点 `6`，饱和到 `6`。

## 规则

- **正半轴网格** `0, 0.5, 1, 1.5, 2, 3, 4, 6`：`0.5` 是唯一的非零次正规（`e=0, m=1` 得
  `m/2 · 2^0`），**不存在 `0.25` 这一档**，所以 `0.1` 只能回到 `0`。
- **最近邻 + RTNE**：恰好落在中点（`0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5.0`）时取
  **尾数位为 0** 的那个码点——注意不是「取偶数幅值」，`1.25` 的中点是 `1.0` 与 `1.5`，
  尾数位 0 的是 `1.0`；`3.5` 的中点是 `3.0` 与 `4.0`，尾数位 0 的是 `4.0`。
- **饱和而非溢出**：`|scaled| > 6` 一律落到 `±6` 并置 `sat_mask`；`6.0` 本身**不**算饱和。
  `inf` 也走饱和路径；`NaN` 无码可编，直接 `ValueError`。
- **符号位独立于数值**：`-0.0` 编码为 `8`（不是 `0`），反量化仍是 `-0.0`；任何绝对值小于
  `0.25` 的负数都回 `-0.0`。
- **两级缩放**：先算 `eff = global_scale * block_scales[b]`（乘积，不是两连除），再算
  `scaled = v / eff`。MXFP4 的 `global_scale` 恒为 `1`（单级），NVFP4 才是两级嵌套。
- **块壁**：`b = i // block_size`，尾块不足 `block_size` 也照自己的 scale 记账；判题会
  用「同值不同块」的用例确认没有串用 scale。
- **校验**：`block_size` 必须是正整数（`bool` 拒绝）、`block_scales` 长度必须等于块数且每项
  有限正数、`global_scale` 有限正数、`values` 不含 `NaN`；空输入合法（块数 0、`block_scales` 空）。

## 提示

- 平局判定要用**精确浮点相等**（中点都是二进制可精确表示的数），不能在反量化之后反推。
- `torch.signbit` / `math.copysign` 才能区分 `±0`，`v < 0` 会漏掉负零。
