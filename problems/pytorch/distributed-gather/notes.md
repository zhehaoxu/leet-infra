# 解题思路

## 面试官意图

Gather / All-Gather、Scatter / Broadcast 这两对「单目标 vs 全目标」的区别，是集合通信最容易被含糊带过的
地方。能立刻答出「Gather 只有一张卡拿到全体、其余卡原样不变」，说明你真理解了通信的**目的端**语义，而不
是把 `all_*` 当成默认。

## 语义

每 rank 交出一片等长数据；拼接后的完整结果**只送到 `dst`**。目标 rank 的缓冲变成 $\bigoplus_q \texttt{inputs}[q]$，
**非目标 rank 保持自己那片不变**（gather 不给它们任何结果）。本题把这个「非目标不变」的 MPI 未定义行为
**规定为原样保留**，使判题确定。

## 与相邻原语的对照（考点）

- **All-Gather**：拼接结果给**所有** rank（Gather 的「广播版」）。
- **Gather**：拼接结果只给 `dst`。
- **Reduce**（对比）：把各 rank 数据**求和**后只给 `dst`——聚合而非拼接；本题是数值拼接、不做加法。

## 实现要点

`torch.cat(inputs, 0)` 得到完整结果，放进 `out[dst]`；其余 `out[j]=inputs[j].clone()`。别把非目标 rank
也填成完整结果——那会变成 All-Gather（`check_only_dst_has_full` 专门卡这点）。`dst` 不一定是 0
（`check_various_dst` 遍历）。用 `.clone()` 保证不共享输入存储。

## 真实场景

`torch.distributed.gather` / NCCL `gather` 常用来把各卡本地指标（loss、grad-norm、token 计数）汇总到 rank0
记录，而无需回传全体；比 All-Gather 少一次广播，省下一份通信。它与 `distributed-scatter` 互为逆：
scatter 把 `dst` 的完整数据切成每 rank 一片，gather 把每 rank 一片收拢回 `dst` 的完整数据。

## 判题口径

`test_cases.py`：样例、只有 dst 拿全体其余原样、dst 遍历含非 0、单 rank、不修改输入；独立参考用逐元素
手动拼接（不走 `torch.cat`）对拍。
