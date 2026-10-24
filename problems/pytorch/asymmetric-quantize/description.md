# 非对称量化（Asymmetric Fake-Quant）

## 题目描述

非对称量化用数据的 $\min/\max$ 定标，让整数范围 $[0,255]$ 对齐浮点 $[\min,\max]$，浮点 0 落在某个非零整数格点（zero_point）。相比对称版，它充分利用格点、更适合**分布不关于 0 对称**的张量（如 ReLU 后的全正激活），代价是要额外存一个 zero_point。

## 规则

- 实现 `solve(x)`，per-tensor uint8 非对称量化并返回**反量化浮点张量**（与 `x` 同形同 dtype）：
  1. $\text{scale} = \dfrac{\max - \min}{255}$
  2. $z = \operatorname{clamp}(\operatorname{round}(-\min/\text{scale}),\,0,\,255)$
  3. $q = \operatorname{clamp}(\operatorname{round}(x/\text{scale}) + z,\,0,\,255)$
  4. 返回 $(q - z)\cdot\text{scale}$
- $\text{scale}=0$（常数张量）时原样返回。
- 结果须与 `torch.fake_quantize_per_tensor_affine(x, scale, z, 0, 255)`（同 scale/z）一致。

## 示例

```
输入：x = [-1.0, 0.0, 0.5, 2.0]
      min=-1, max=2 → scale = 3/255 ≈ 0.011765, zero_point = 85
输出：[-1.0, 0.0, 0.49412, 2.0]
（min/max 命中格点边界精确还原；0.5 落到 42 格 → (42+85-85)*scale ≈ 0.49412）
```

## 约束

- 支持 float32 / float64；输出 dtype 与输入一致
- `round` 用就近取整（银行家舍入，与 `torch.round` 一致）
- 不使用 `fake_quantize_*`；需手写 scale/zero_point 估计 + round/clamp/dequant
