# 解题说明 · Merging

## 核心 5 步

1. topk 选 kept、nonzero 补集拿 evict
2. 两侧 key 归一化 → 余弦相似度矩阵
3. 每 evict 找最相似 kept slot（max-sim）
4. 权重 = 相似度 × 范数比
5. scatter_add 到目标 slot → 归一化 → 写回

## 为什么用 norm 比

单纯按 sim 加权会让"大 value 把小 value 拉偏"——例如 $\|v_\text{evict}\| = 100, \|v_\text{target}\| = 1$ 时，合并结果会被 evict 淹没。范数比 $e / (e + t)$ 相当于一个 **软 sigmoid**：evict 相对更弱时权重高（把弱邻居信息聚合过来），evict 更强时权重被稀释（避免主导）。

## similarity_threshold 与 merge_fraction

真实 kvpress 允许两个额外门：
- `similarity_threshold > 0`：只合并相似度够高的
- `merge_fraction < 1`：只合并 top-$\alpha$ 相似度的 evict

题目固定 `threshold=0.0, fraction=1.0` 简化——但仍保留 `max_sim.clamp(min=0)`：负 sim 直接被丢弃（等价 threshold=0）。

## 只改 kept 位、不动 evict 位

`result = values.clone(); result.scatter_(1, keep_idx, updated)` —— evict 位置的 value 数值上"还在原地"但会被下游 topk 过滤掉。题目返回 full-shape，判题时 kept 位比对合并结果、evict 位比对原值。

## 与 CAM / D2O 的关系

CAM 也是"合并代替剪枝"，用 attention-based 加权；D2O 用 output deviation 决定合并目标。Merging 用 **cos 相似度**——最简单、也是这个家族的原型方法。
