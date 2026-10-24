# LLM.int8()：离群列混合精度拆分

## 题目描述

**LLM.int8()**（Dettmers et al., 2022, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)）是最早的" emergent feature dimension（涌现特征维）"处理方案：矩阵乘里极少数**输入通道**会持续出现超大激活，把它们的量化误差放大到破坏精度。做法是把权重沿输入通道切成两半——**离群列**用 FP16 全精度保留、其余"普通列"用 INT8 量化——两半分别做矩阵乘再相加。等价于：把离群列的权重**原样**留下、只对其余列量化。

## 规则

给定权重 $W$ 形状 $(O, C)$、离群阈值 $\tau > 0$、位宽 `bits`（默认 8）。

1. **离群列判定**：第 $j$ 列为离群列，当且仅当该列存在某个 $o$ 使 $|W_{oj}| > \tau$，即 $\max_o |W_{oj}| > \tau$。得到 `outlier_mask` 形状 $(C,)$（`bool`）。
2. **量化普通列**：对所有**非离群列**做 **per-tensor 对称 RTN**——量程 scale $=\max\{|W_{oj}|: j\ \text{非离群}\}$，$q_{max}=2^{bits-1}-1$，$q=\operatorname{round}(W/\text{scale}\cdot q_{max})$ 截断到 $[-q_{max}-1,\,q_{max}]$，反量化 $q/q_{max}\cdot\text{scale}$。
3. **离群列原样保留**（视作 FP 全精度）。

返回 `(W_recon, outlier_mask)`，$W_{recon}$ 形状 $(O,C)$：离群列等于原始 $W$、普通列为其反量化值。

## 示例

```
W = [[0.1, 0.2, 9.0, 0.3],
     [0.4, 0.5, 8.0, 0.6],
     [0.7, 0.8, 7.0, 0.9]]
tau = 6.0, bits = 8

列 absmax: [0.7, 0.8, 9.0, 0.9]  → 仅 col2 > 6 → outlier_mask = [F, F, T, F]
普通列 {0,1,3} 的 scale = max 普通列 absmax = 0.9
col2 原样：[9.0, 8.0, 7.0]
其余列 8-bit 量化（几乎无损）：
W_recon ≈ [[0.0992, 0.1984, 9.0, 0.2976],
           [0.3969, 0.5031, 8.0, 0.6024],
           [0.7016, 0.8008, 7.0, 0.9000]]
```

## 约束

- CPU float32；`outlier_mask` 用 `torch.equal` 比较、`W_recon` 用 `allclose(atol=1e-4)`。
- 若全部列都是离群列，则无普通列可量、`W_recon` 等于 `W`；若 scale 为 0（普通列全 0）按 1 处理。
- 取整用就近、`.5` 取偶（与 `torch.round` 一致）。
