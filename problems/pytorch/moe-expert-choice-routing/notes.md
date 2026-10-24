# 解题说明 · Expert-Choice 路由

## 关键洞察：转置方向

- 传统 top-k：每 token 挑 $k$ 个专家 → 每 token 恰好 $k$ 次分派，但每专家负载不均。
- EC：每专家挑 $k_c$ 个 token → 每专家恰好 $k_c$ 负载，但每 token 分派数不定（0/1/2/…）。

EC 的容量分配**天生完美均衡**，无需 aux loss。代价：某些 token「无家可归」（没被任何专家挑中），需要兜底处理。

## tie-break

`torch.topk(probs[:, e], k_c)` 对相同 probs 的 token 顺序不保证；测试用例避免严格并列。若需要 tie-break by token 索引升序，用 `sorted(range(T), key=lambda t: (-probs[t,e], t))[:k_c]`。

## renorm 分母

`weights.sum(-1, keepdim=True)` 可能为 0（某 token 一列都没选中）。用 `torch.where(sum>0, w/sum, 0)` 显式置零，避免 `nan = 0/0`。

## 每列恰 $k_c$

这是 EC 的核心保证：`mask[:, e].sum() == k_c` for all e。测试断言该不变式。

## 总激活参数

若 $E \cdot k_c = T \cdot k$，则平均每 token 被 $k$ 个专家挑中——但**具体每个 token 被几个是不定的**，某些 0、某些大于 $k$。
