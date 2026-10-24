# 对称量化（Symmetric Fake-Quant）

## 题目描述

对称量化把浮点张量映射到整数格点（如 int8），量化零点固定为 0、量程关于原点对称。它是量化感知训练（QAT）里的核心构件：**fake-quant** = 量化 → 反量化走一遍，返回「被量化噪声污染后仍是浮点」的结果，梯度可直通。

## 规则

- 实现 `solve(x)`，per-tensor 对称量化并返回**反量化后的浮点张量**（与 `x` 同形同 dtype）：
  1. $\text{scale} = \dfrac{\max_i |x_i|}{127}$
  2. $q_i = \operatorname{clamp}(\operatorname{round}(x_i/\text{scale}),\,-127,\,127)$
  3. 返回 $q_i \cdot \text{scale}$
- $\text{scale}$ 恒正；全零张量原样返回（避免除零）。
- 结果须与 `torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)`（同 scale）一致。

## 示例

```
输入：x = [0.2, -1.0, 0.7, -0.3]
      absmax = 1.0 → scale = 1/127 ≈ 0.0078740
      格点整数 q = [25, -127, 89, -38]
输出：[0.19685, -1.0, 0.70079, -0.29921]
（-1.0 命中最大格点原样保留；0.2 落到 25 格 → 25/127 ≈ 0.19685）
```

## 约束

- 支持 float32 / float64；输出 dtype 与输入一致
- `round` 用就近取整（银行家舍入，与 `torch.round` 一致）
- 不使用 `fake_quantize_*`；需手写 scale 估计 + round/clamp/dequant 三步
