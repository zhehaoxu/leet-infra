# Merging：驱逐 token 的 value 折进最相似邻居

## 题目描述

普通 top-k 剪枝把低分 token 的 value **硬扔**——信息永久丢失。**MergingPress** 提出软落地：把每个被驱逐 token 的 value **加权合并** 到"最相近的存活者"身上（相似度用 key 的余弦），保留信息但形状不变。灵感来自 Token Merging (ICLR 2023) 与 D2O。

合并权重 $w$ 由两部分构成：
- **相似度门**：$\max(\text{sim}, 0)$（若最近邻余弦为负则丢弃）
- **范数比**：$\|v_\text{evict}\| / (\|v_\text{evict}\| + \|v_\text{target}\| + \varepsilon)$ —— 大 value 合并进小 value 时权重高，反之被稀释

目标 kept slot 的新 value 用 **加权平均**：$v_\text{new} = (v_\text{kept} + \sum w \cdot v_\text{evict}) / (1 + \sum w)$。若某 kept slot 未被任何 evict 选中，保持原值不变。

## 规则

给定 `keys` $(H, L, D)$、`values` $(H, L, D)$、`scores` $(H, L)$、`n_kept`：

1. `kept_idx = scores.topk(n_kept, largest=True).indices` — shape $(H, n\_kept)$，**按分数降序排列**
2. `evict_idx` 为 kept 的补集，按原序列顺序 $(H, L - n\_kept)$
3. 归一化两组 key（除 L2 norm，clamp min = $\varepsilon$）
4. `sim = ev_keys_normalized @ kept_keys_normalized^T` → $(H, n\_evict, n\_kept)$
5. 对每个 evict 挑 `max_sim, target_slot`（沿 $n\_kept$ 取 max）
6. $w = \max(\text{sim}, 0) \cdot \dfrac{\|v_\text{evict}\|}{\|v_\text{evict}\| + \|v_\text{target}\| + \varepsilon}$
7. `vacc[slot] += w * v_evict`；`wacc[slot] += w`（scatter_add）
8. `updated[slot] = (v_kept[slot] + vacc[slot]) / (1 + wacc[slot])` 若 `wacc[slot] > 0`；否则 `v_kept[slot]`
9. **scatter 回原序列**：只改写 kept 位置的 value，evicted 位置的 value 保留原值（不变）

返回 shape $(H, L, D)$ 的 float32 `merged_values`。固定 $\varepsilon = 10^{-6}$、`similarity_threshold = 0`、`merge_fraction = 1`。

## 示例

```
H=2, L=4, D=2, n_kept=2
scores = [[0.9, 0.1, 0.7, 0.2],   [0.3, 0.4, 0.8, 0.1]]

h=0: kept_idx=[0, 2]（分 0.9, 0.7），evict=[1, 3]
  两个 evict 都最相似 kept slot 0 → slot 0 value 被改写，slot 1 保持
  merged values[0] = [1.814293, 2.814292]
  merged values[2] = [5.717330, 6.717330]

h=1: kept_idx=[2, 1]（分 0.8, 0.4），evict=[0, 3]
  两个 evict 都最相似 kept slot 1（对应原位置 1）→ slot 0 保持不变
  merged values[2] = [2, 3]（无贡献）
  merged values[1] = [1.821373, 2.124212]

result = [[[1.814293, 2.814292], [3.0, 4.0], [5.717330, 6.717330], [7.0, 8.0]],
          [[0.5, 0.5], [1.821373, 2.124212], [2.0, 3.0], [4.0, 5.0]]]
```

## 约束

- CPU float32；$\varepsilon = 10^{-6}$。
- `torch.topk` 输出按分数降序、tie 按索引升序。
- `evict_idx` 顺序即原位置升序（`evict_mask.nonzero()[:, 1].reshape(H, n_evict)`）。
