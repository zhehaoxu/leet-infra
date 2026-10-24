# 按通道量化（Per-Channel Fake-Quant）

## 题目描述

per-tensor 对称量化（见 `symmetric-quantize`）用**一个** scale 覆盖整个张量。卷积/线性权重往往各输出通道动态范围差异巨大——一个离群通道会把 scale 撑大、把其余通道的分辨率全压没。**按通道量化（per-channel / per-axis）**给每个通道一套独立 scale，是权重量化的事实标准。本题实现它的 fake-quant 版本。

## 规则

实现 `solve(x, axis)`，沿 `axis` 做对称 per-channel fake-quant，返回**反量化后的浮点张量**（与 `x` 同形同 dtype）：

1. 对每个通道 $c$（`axis` 上的一个索引），求该通道**其余所有维度**上的 $\text{absmax}_c = \max_{i \notin \text{axis}} |x|$。
2. $\text{scale}_c = \text{absmax}_c / 127$。
3. $q = \operatorname{clamp}(\operatorname{round}(x / \text{scale}_c),\,-127,\,127)$，返回 $q \cdot \text{scale}_c$。
4. **退化通道**：若某通道全零（$\text{absmax}_c = 0$），该通道原样返回 0（避免除零，且必须与 torch 内置 oracle 一致）。

> `axis` 给的是「通道维」。scale 向量长度 = `x.size(axis)`，广播到其余维。例如权重 `[out, in]`、`axis=0` → 每个 `out` 行一套 scale，absmax 沿 `in` 维归约。

## 示例

```
x    = [[1.0, -4.0, 2.0, 0.5],
        [0.0,  0.0, 0.0, 0.0],
        [3.0,  1.5,-6.0,-0.75]],  axis = 0

各通道 absmax = [4.0, 0.0, 6.0]  →  scale = [4/127, 0, 6/127]
输出 = [[1.00787, -4.0, 2.01575, 0.50394],
        [0.0,      0.0, 0.0,     0.0     ],     # 全零通道原样 0
        [3.02362,  1.51181, -6.0, -0.75591]]
```

（第 0 行 absmax=4 命中格点，`-4.0` 精确还原；第 2 行 `3.0→round(3/scale)=80→80*scale≈3.0236`）

## 约束

- 支持任意维 `x`（≥1 维）与任意合法 `axis`（可为负）；`round` 用就近取整（银行家舍入，同 `torch.round`）
- 结果须与 `torch.fake_quantize_per_channel_affine(x, scales, zeros, axis, -127, 127)`（`scales = absmax/127`、`zeros = 0` int32）一致——这是本题的内置 oracle
- 不使用 `fake_quantize_*`，手写 absmax/scale/round/clamp/dequant
