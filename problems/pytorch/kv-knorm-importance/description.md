# KNorm：按 key L2 范数打重要性分

## 题目描述

**KNorm**（Zhang et al., 2024, [arXiv:2406.11430](https://arxiv.org/abs/2406.11430)）：观察「attention 得分大 ↔ key 的 L2 范数大」，因此直接用 $-\|k\|_2$ 做重要性分——分数越大越保留，即 **norm 越大越重要**。这是最便宜、无需 attention 的 KV 打分。

kvpress 里 `KnormPress.score` 就一行：`return -torch.linalg.vector_norm(keys, dim=-1, dtype=promote(dtype, float32))`——为避免 fp16/bf16 打平，先在 float32 上累加。

## 规则

实现 `solve(keys)`，`keys` 形状 $(H, L, D)$（无 batch，float32）：返回 shape $(H, L)$ 的分数向量 $-\|k_{h,\ell}\|_2$。

## 示例

```
keys = [
  h=0: [[1,2,2], [0,0,0], [3,4,0]],   # 范数 [3, 0, 5]
  h=1: [[0.5,0.5,0.5], [1,1,1], [2,0,0]]  # 范数 [√0.75, √3, 2]
]
scores = [
  [-3.0,        -0.0,       -5.0],
  [-0.86602538, -1.7320508, -2.0]
]
```

## 约束

- 输入已 float32，返回 float32；不需要显式 dtype 提升。
- $\|0\|_2 = 0$，取负即 `-0.0`；与 `0.0` 数值相等。
