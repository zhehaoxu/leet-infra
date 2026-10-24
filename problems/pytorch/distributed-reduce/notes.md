# Reduce（根归约）详解

## Reduce 在集合通信家族里的位置

把「谁最终拿到结果」作为分类轴，几个归约类原语一目了然：

| 原语 | 结果落点 | 每 rank 结果 |
|---|---|---|
| **Reduce** | 只有 root | root 拿完整和，其余**不变** |
| **All-Reduce** | 所有 rank | 人人拿完整和 |
| **Reduce-Scatter** | 所有 rank，但各留一段 | 每人拿 $1/W$ 的局部和 |
| Scatter/Gather | 一对多 / 多对一 | 分发 / 收集 |

关系式：`All-Reduce = Reduce + Broadcast`；`All-Reduce = Reduce-Scatter + All-Gather`。本题专做最原始的那一步——**归约到根**。

## 建模要点

- 纯函数视角：入参是通信前各 rank 的本地张量列表 + root 下标，出参是通信后列表。
- **非根 rank 保留自己的原始输入**（不是零、不是和）。这是 Reduce 区别于 All-Reduce 的语义核心，也是最容易被写成「广播给所有人」的坑——`check_only_root_has_sum` 会逐个校验非根项仍等于其输入。
- root 项是**完整**逐元素和（形状与输入一致），而非某一分片。

## 实现坑

- **别就地改 inputs**：`out[root] = total` 前要先把列表复制成 `[t.clone() for t in inputs]`，否则会污染输入（`check_input_not_mutated` 抓这个）。
- 求和路径随意（`stack().sum()` 或顺序累加），浮点用 `allclose` 判等；整型可 `torch.equal` 精确判。
- `W=1` 时 root 唯一，结果即其自身——退化情形要成立。

## 关联

- 上游/对照：`distributed-all-reduce`（多一步广播）、`distributed-reduce-scatter`（结果分片）。
- 下游：`distributed-collective-latency` 分析这类归约在 ring / recursive-doubling 下的耗时。
- 判题：CPU 纯张量模拟 + `test_cases` 独立重算核对，无 GPU/多进程依赖。
