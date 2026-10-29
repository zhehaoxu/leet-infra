# weight_norm 重参数化反向：dg 与 dv 的投影消歧

## 题目描述

`weight_norm` 把权重写成 \(W=g\cdot V/\lVert V\rVert\)（沿第 0 轴逐切片），好处是「方向」与「长度」解耦，代价是 \(g\) 与 \(\lVert v\rVert\) 这两个自由度**不可辨识**：把 \(v_i\) 整体缩放 \(c\) 倍、\(g_i\) 不变，\(W\) 完全不变。反向因此必须把 \(dW\) 沿单位方向 \(\hat v\) 的分量**投影掉**再给 \(dV\)，否则那点自由度会被 \(dg\) 和 \(dV\) 各算一次。本题给你 `V, g, dW`，返回 `(dg, dV)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- \(\hat v_i=v_i/\lVert v_i\rVert_2\)，\(dg_i=\sum_j dW_{i,j}\hat v_{i,j}\)（\(dW\) 在该切片单位方向上的投影）。
- \(dV_i=(g_i/\lVert v_i\rVert)\big(dW_i-\hat v_i\,dg_i\big)\)：**必须先减去沿 \(\hat v\) 的分量**。
- 两条硬自检：\(\sum_j dV_{i,j}v_{i,j}=0\)（\(dV\) 与 \(v\) 正交）；把 \(V\) 的第 \(i\) 片缩放任意 \(c\ne0\)（\(g,dW\) 不变）时 \(W\) 与 \(dg\) 都不变。
- `V` 至少二维、`g` 长度等于第 0 轴、`dW` 与 `V` 同形；出现零向量切片（方向无定义）一律抛 ValueError。

## 示例

两片二维向量、\(dW=I\)，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
V = [[3, 4],     g = [2, 1]      dW = [[1, 0],
     [0, 5]]                            [0, 1]]
‖v‖ = [5, 5],  v̂ = [[0.6, 0.8], [0, 1]]
dg = [0.6, 1.0]                       # dg_i = dW_i · v̂_i
dV = [[0.256, -0.192], [0, 0]]        # (g/‖v‖)·(dW − v̂·dg)
```

第 0 片：\(0.4\cdot([1,0]-0.6[0.6,0.8])=0.4[0.64,-0.48]\)；第 1 片 \(dW\) 与 \(\hat v\) 共线，减完正好是 0。

## 约束

- 内部 float64 计算，`dg` 的 dtype 随 `g`、`dV` 随 `V`；`V` 可为 `(out, in)` 或 `(out, in, kh, kw)`
- 与 `torch.nn.utils.weight_norm` 等价前向（`W = g·V/‖V‖`）的 `torch.autograd.grad` 对拍，float64 atol=1e-9
