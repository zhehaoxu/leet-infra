# KV Cache：CriticalKV 两阶段 Wo·V 加权保留

## 题目描述

H2O / SnapKV 一系的打分只看「key 收到多少 attention」，但一个 token 是否**真正关键**还取决于它的 **value 内容**穿过注意力输出投影 \(W_o\) 后贡献多大。CriticalKV（Li et al., 2025, [arXiv:2502.03805](https://arxiv.org/abs/2502.03805)；kvpress `NVIDIA/kvpress` 的 `kvpress/presses/criticalkv_press.py` 中 `CriticalKVPress`）把任意基础打分器包装成**两阶段选择**：

- **阶段 1**：按基础分先圈出一小批「受保护」token；
- **阶段 2**：其余候选改用新分数——基础分（加平滑项）乘以该 token 的 value 经输出投影后的 L1 范数 \(\|(v_t W_o)\|_1\)；
- 阶段 1 圈中的位置在最终分数里被填成该数据类型的最大浮点数（保底分），保证它们在阶段 2 一定幸存。

## 规则

输入：每个 kv head 的基础分数 \(H\times L\)；同形状的 `wov_norm`，即每个 token 的 \(\|v_t W_o\|_1\) 在一个 GQA 组内各 query head 上的平均值；标量 `compression_ratio` \(r\in[0,1)\)、`first_stage_ratio` \(s\in[0,1]\)、`epsilon` \(\varepsilon\ge 0\)。

1. 总保留量 \(n\_kept=\max(1,\lfloor L(1-r)\rfloor)\)。
2. 阶段 1 预算 \(b_1=\lfloor L(1-r)\,s\rfloor\)；若 \(s>0\) 则 \(b_1=\max(1,b_1)\)（短上下文也至少保 1 个，这是上游代码的防呆设计），且 \(b_1\le L\)。
3. 阶段 1：每行按基础分取 top-\(b_1\)（分大者优先，**平局取位置小者**）。
4. 阶段 2 最终分：\(\text{final}=(\text{base}+\varepsilon)\cdot \text{wov\_norm}\)，然后把阶段 1 命中的位置改写为 float32 上限 \(3.4028235\times10^{38}\)。
5. 每行按最终分取 top-\(n\_kept\)（平局取位置小者），把保留位置**升序**输出。

返回两个张量：`kept`（int64，形状 \((H, n\_kept)\)）与最终分数矩阵（float32，形状 \((H, L)\)）。

## 示例

```
base row0 = [0.9, 0.2, 0.7, 0.1, 0.5, 0.3]   wov_norm row0 = [1.0, 3.0, 0.5, 2.0, 4.0, 0.2]
base row1 = [0.4, 0.8, 0.1, 0.6, 0.2, 0.9]   wov_norm row1 = [2.0, 0.1, 5.0, 0.5, 1.0, 0.3]
r = 0.5, s = 0.5, eps = 1e-4  →  n_kept = 3, b1 = 1

row0：阶段1 命中位置 {0}；
  final = [max, 0.6003000, 0.3500500, 0.2002000, 2.0004001, 0.0600200]   (max = 3.4028235e+38)
  top-3 → kept = [0, 1, 4]
row1：阶段1 命中位置 {5}；
  final = [0.8002000, 0.0800100, 0.5005000, 0.3000500, 0.2001000, max]
  top-3 → kept = [0, 2, 5]
```

## 约束

- CPU float32；`wov_norm` 非负；\(0\le r<1\)，\(0\le s\le 1\)。
- **不变量**：阶段 1 集合 ⊆ 最终保留集合（\(b_1\le n\_kept\) 恒成立）。
- \(s=0\) 时退化为纯 value 投影加权 top-k（无任何保底）；\(r=0\) 时每行保留全部 \(L\) 个位置。
- 所有平局一律按「位置小者优先」打破，输出前升序排序。
