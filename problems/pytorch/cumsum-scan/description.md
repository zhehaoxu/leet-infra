# 累积和扫描（Cumsum / Prefix Scan）

## 题目描述

累积和（前缀和）沿某个轴输出「到目前为止的总和」：第 `i` 个位置等于轴上 `0..i` 的累加。
它是 scan 类算子中最基本的一种，广泛用于把归约变成「保留每一步中间结果」——例如
top-p 采样里对概率做前缀和找截断点、Transformer 相对位置/因果掩码构造、以及
linear-attention / SSM（Mamba）里把递归状态写成并行 scan。请**不用 `torch.cumsum`** 实现。

## 规则

- 实现 `solve(x, axis)`，返回沿 `axis` 的累积和，形状与输入相同。
- `axis` 可为负。
- 递推定义：`out[0] = x[0]`，`out[i] = out[i-1] + x[i]`（沿 `axis`）。
- 结果须与 `torch.cumsum(x, dim=axis)` 一致。

## 示例

```
输入：x = [[1, 2, 3],
          [4, 5, 6]]
  axis=1（每行内前缀和）→ [[1, 3, 6],
                          [4, 9, 15]]
  axis=0（每列跨行前缀和）→ [[1, 2, 3],
                            [5, 7, 9]]
```

## 约束

- 支持任意维输入；沿轴 `axis` 累加
- 不得使用 `torch.cumsum` / `torch.addcmul` 等现成 scan 算子
