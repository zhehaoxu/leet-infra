# 全连接线性层（Linear / Affine Layer）

## 题目描述

线性层是最常用的网络构件：$y = xW^\top + b$。别被 `nn.Linear` 的一行调用迷惑——
它内部是「每个输出单元 = 输入向量与该单元权重行的点积，再加偏置」。理解它的
**权重方向约定**（`weight` 形状是 `(out_features, in_features)`，不是转置的那份）是
读代码、调梯度、写融合 kernel 的第一课。

## 规则

- 实现 `solve(x, weight, bias=None)`，`x` 形状 `(..., in_features)`，
  `weight` 形状 `(out_features, in_features)`，`bias` 形状 `(out_features,)`。
- 输出形状 `(..., out_features)`：对输入的最后一维与每个 `weight[o]` 做点积。
- 支持 `x` 含任意前置 batch 维（如 `(N, L, in)`）。
- 结果须与 `F.linear(x, weight, bias)`（即 `x @ weight.T + bias`）一致。

## 示例

```
输入：x = [[1, 2, 3],
          [4, 5, 6]]
     weight = [[1, 0, 2],
               [-1, 3, 0]]，bias = [10, -1]
  行0：[1·1+2·0+3·2+10, 1·(-1)+2·3+3·0-1] = [17, 4]
  行1：[4·1+5·0+6·2+10, 4·(-1)+5·3+6·0-1] = [26, 10]
输出：[[17, 4],
      [26, 10]]
```

## 约束

- float32 输入；`bias` 可为 `None`
- 重点掌握 `weight` 的 `(out, in)` 存储约定与「点积 + 偏置」的语义
