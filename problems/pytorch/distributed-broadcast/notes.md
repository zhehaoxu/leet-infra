# 解题思路

## 面试官意图

Broadcast 是集合通信里最简单却无处不在的一个：DDP 初始化时把 rank0 的模型参数广播到所有卡、
`torch.distributed.broadcast` 同步 loss scaler、下发一个标量配置。理解它，才能把它和 All-Gather
区分开——两者都是「让所有 rank 拿到东西」，但数据来源完全不同。

## 语义

指定源 rank `src`，只有 `inputs[src]` 是有效数据，其余 rank 的输入是占位。通信后**每个 rank 都等于
`inputs[src]` 的一份拷贝**。注意它是**一对多复制**，不做任何合并或拼接。

## 与 All-Gather / Scatter 的对照

- **Broadcast**：一份数据 → 所有 rank（同一份，重复）。
- **All-Gather**：每 rank 一片不同的数据 → 所有 rank 拿到「拼接后的全体」。
- **Scatter**（见 `distributed-scatter`）：src 持有完整数据 → 拆成 $W$ 片分给各 rank（Broadcast 的「切片版」）。

## 实现要点

直接取 `inputs[src]`、复制 $W$ 份即可。用 `.clone()` 保证不同 rank 之间、以及与输入之间不共享存储，
从而满足「不修改输入」。`src` 可以不是 0，别硬编码 `inputs[0]`——这是 `check_src_index_used` 的考点。

## 真实通信量

Broadcast 一份大小 $B$ 的数据，朴素做法是 src 连发 $W-1$ 次共 $(W-1)B$；NCCL 用树形广播把**延迟**
降到 $O(\log W)$（每层并行转发），带宽利用率更高。这也是大集群下发权重不会成为瓶颈的原因。

## 判题口径

`test_cases.py` 校验：正确取用 `src` 索引（含 src≠0）、二维与不同占位长度、单 rank 退化、不修改输入。
