# PEFT：EVA 解释方差驱动的全局秩分配

## 题目描述

**EVA**（Explained Variance Adaptation，arXiv:2410.07170，ELLIS Unit LIT AI Lab，已入 PEFT PR）把 LoRA 的秩预算**跨层统一调度**：不再给每层固定 rank，而是对每层输入做增量主成分分析，按**解释方差**把全局预算 `len(层) × rank` 个分量分给"最值得"的层——方差大的层多拿秩，方差小的层可以拿 0。

官方 ml-jku/EVA 实现（`src/svd.py::compute_svd`）的确定性账（锚点全部可离线重算）：

```
# 1) 指标四态（_get_metric）
raw   = explained_variance_
ratio = ev / ev.sum()        # 每层各自归一！
sum   = ev / ev.sum()
max   = ev / ev.max()

# 2) 全局秩分配（_get_rank_distribution）
ev = {层: metric(层向量)[: round(rank * rho)]}      # 只取前 rank*rho 个分量
扁平 (层, 值) 全对 → argsort 降序 → 前 len(层)*rank 个（全局预算）逐项投票
counts = Counter(投票的层)；没拿到票的层显式补 0
# equal-inputs 修正：等价层（SVD 从别层复制）的 SVD 源层秩若更低则交换
# （保证算过 SVD 的层秩 >= 等价层；单向交换，已 >= 则跳过）

# 3) 组装（svd_dict）
u = components[源层][:rank_dist[层]]                # 秩为 0 的层整体跳过
whiten: u /= sqrt(singular_values)                  # else（elif！）
scale_by_singular_values: s /= s.max(); u *= s      # 双 True 时 whiten 优先
```

三个关键口径：

1. **每层先各自归一再跨层排序**：`ratio`/`sum`/`max` 都是**层内**归一——绝对值小的层归一后可能反超（例：某层 ev=[0.2,0.1,0.08,0.04,0.03,0.01] 总和 0.46，归一后首分量 0.435 超过另一层总和为 1 的第二分量 0.35）。`raw` 则保持绝对值排序。
2. **截断在归一之后、投票之前**：每层只取前 `round(rank·rho)` 个分量参与投票（`rho≥1`，官方 assert 保护）；全局预算 `len(层)·rank` 截断在扁平序列上做。
3. **whiten 与 scale 互斥（elif）**：`whiten=True` 时除以 `sqrt(s)`，否则 `scale_by_singular_values=True` 时乘 `s/max(s)`；两个都设时 whiten 赢（官方 `elif`）。

## 任务

实现四组入口（判题域 `explained_variance` 向量已离线算好，显式入参）：

- `mode="metric"`：`explained_variance` 一维张量 + `metric` ∈ {raw, ratio, sum, max} → 归一后向量；非法 metric 抛 `ValueError`。
- `mode="rank_dist"`：`exp_vars`（层名 → 向量 dict）、`hook_layer_map`（层名 → SVD 源层名 dict）、`equal_inputs_map`（等价层 → 源层 dict）、`rank`、`rho`、`metric` → `{层名: 秩}` dict；`rho < 1` 抛 `AssertionError`（官方裸 assert）。
- `mode="scale_components"`：`components` (n, d)、`singular_values` (n,)、`whiten`、`scale_by_singular_values` → 缩放后 (n, d)；两开关都 False 时原样返回。
- `mode="svd_dict"`：`components`、`singular_values`（层名 → 张量 dict）、`hook_layer_map`、`rank_dist` → `{层名: u}`；秩 0 层不进 dict；等价层切**源层**的 components 并按**源层**的 singular_values 缩放。

## 示例

三层 `rank=2, rho=2`（max_components=4，预算 6），metric=ratio：

```
l0 ev = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]   (sum=1.0)  → [0.5, 0.3, 0.1, 0.05]
l1 ev = [0.4, 0.35, 0.15, 0.06, 0.02, 0.02] (sum=1.0)  → [0.4, 0.35, 0.15, 0.06]
l2 ev = [0.2, 0.1, 0.08, 0.04, 0.03, 0.01]  (sum=0.46) → [0.435, 0.217, 0.174, 0.087]
扁平降序 top6: 0.5(l0), 0.435(l2), 0.4(l1), 0.35(l1), 0.3(l0), 0.217(l2)
→ {l0: 2, l1: 2, l2: 2}     # l2 绝对值最小却拿满预算的一半
```

`scale_components(u, s=[4,1,0.25], whiten=True, scale_by_singular_values=True)` → `u / sqrt(s)`（whiten 优先）。

## 约束

- 全程 float64；argsort 用 torch（降序）；Counter 缺票层显式补 0（int）。
- `equal_inputs_map` 交换只在源层秩**严格更低**时发生（相等不交换）；判题域内展开值相同故该交换以 no-op 为主，但仍按官方实现写全。
- `svd_dict` 返回的 dict 只含秩 > 0 的层；各层返回独立张量（不共享存储别名断言除外）。
