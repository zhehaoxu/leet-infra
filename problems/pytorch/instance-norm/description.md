# InstanceNorm

## 题目描述

Instance Normalization 对**每个样本的每个通道单独**在空间维上归一化——风格迁移里抹掉每张图自己的对比度信息的标准组件：

$$
y = \frac{x - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}\,\gamma_c + \beta_c,\qquad
\mu_{n,c},\sigma_{n,c}^2 = \text{mean/var over spatial dims } (2,3,\dots)
$$

与 BatchNorm 的关键区别：统计**不跨样本、不跨通道**；与 LayerNorm 的区别：归约的是空间维而不是特征维。

## 规则

- 实现 `solve(x, weight=None, bias=None, eps=1e-5)`，x 形状 `(N, C, *)`（如 `(N,C,H,W)` 或 `(N,C,L)`）。
- 统计维是 `dim >= 2` 的所有空间维，`keepdim=True` 后广播。
- `weight`/`bias` 可为 `None`（等价于恒等仿射）；非 `None` 时形状为 `(C,)`，按通道广播。
- 方差用**有偏**口径；结果与 `F.instance_norm(x, None, None, w, b, use_input_stats=True)` 一致。

## 示例

```
输入：x = [[[1,2],[3,4]], [[5,6],[7,8]]]   # (N=2, C=2, L=2)，无仿射参数
样本0通道0：[1,2] → μ=1.5, σ²=0.25
输出：[[[-0.99998, 0.99998], [-0.99998, 0.99998]],
       [[-0.99998, 0.99998], [-0.99998, 0.99998]]]   # 每个 (样本,通道) 各自归一化
```

## 约束

- float32/float64；每个 (样本, 通道) 的空间元素数 ≥ 2
- 常数通道（σ²=0）不得产生 NaN
