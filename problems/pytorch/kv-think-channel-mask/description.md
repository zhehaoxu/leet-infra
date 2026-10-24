# ThinK：Key 的 head_dim 通道剪枝

## 题目描述

**ThinK**（[arXiv:2407.21018](https://arxiv.org/abs/2407.21018)）压的不是 **token 维**（序列长度），而是 **channel 维**（`head_dim`）：每个 head 内部，某些维度对 attention 得分贡献极小、可以整段丢弃。它与 SnapKV/KNorm 之类"沿 token 剪"正交、可级联。

kvpress 打分公式（**通道 $d$ 的重要性 = 该通道上 query 的均方 × 该通道上 key 的均方**）：

$$ s_{h,d} = \Big(\frac{1}{W}\sum_{w=1}^{W} q_{h,w,d}^2\Big) \cdot \Big(\frac{1}{L}\sum_{\ell=1}^{L} k_{h,\ell,d}^2\Big) $$

其中 $q_{h,w,d}$ 是**已 RoPE 后**的窗口内第 $w$ 个 query 的第 $d$ 维。$n_\text{pruned} = \lfloor D \cdot R\rfloor$ 个最小的通道被**置 0**（保留 shape、只是清零，与"直接扔维"内存等价）。

## 规则

给定 `queries` $(H, W, D)$、`keys` $(H, L, D)$、`compression_ratio` $R \in [0, 1)$：

1. `q_norm = (queries ** 2).mean(dim=1)` → $(H, D)$
2. `k_norm = (keys ** 2).mean(dim=1)` → $(H, D)$
3. `key_scores = q_norm * k_norm` → $(H, D)$
4. `n_pruned = int(D * R)`
5. 对每 head 挑 `key_scores` **最小**的 $n_\text{pruned}$ 通道，`keys` 上对应通道**置 0**

返回 shape $(H, L, D)$ 的 masked_keys。

## 示例

```
H=2, W=2, D=4, L=3, compression_ratio=0.5  → n_pruned=2

queries[h=0] = [[1.0, 0.5, 0.0, 0.5],   queries[h=1] = [[0.0, 0.0, 1.0, 1.0],
              [0.5, 1.0, 0.5, 0.0]]                     [1.0, 0.5, 0.5, 0.0]]

keys[h=0] = [[1,2,3,4],[0.5,0.5,0.5,0.5],[-1,1,-1,1]]
keys[h=1] = [[2,0,0,4],[0,0,1,0],[1,1,2,3]]

h=0: q_norm=[0.625,0.625,0.125,0.125]·k_norm=[0.75,1.75,3.4167,5.75]
     key_scores=[0.46875, 1.09375, 0.42708, 0.71875]
     最小 2 → 通道 {2, 0} → 置 0
h=1: q_norm=[0.5,0.125,0.625,0.5]·k_norm=[1.6667,0.3333,1.6667,8.3333]
     key_scores=[0.8333, 0.0417, 1.0417, 4.1667]
     最小 2 → 通道 {1, 0} → 置 0

masked_keys[h=0] = [[0,2,0,4],[0,0.5,0,0.5],[0,1,0,1]]
masked_keys[h=1] = [[0,0,0,4],[0,0,1,0],[0,0,2,3]]
```

## 约束

- CPU float32；kv-group 视为 1（queries 头数 = keys 头数）。
- `topk(largest=False)` 挑最小；tie 按索引升序。
