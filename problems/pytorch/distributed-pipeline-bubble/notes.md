# 解题思路

## 面试官意图

「流水线并行为什么要求 micro-batch 数远大于 stage 数？」「为什么 p=4、m=4 只有 4/7 的利用率？」——气泡率
$\frac{p-1}{m+p-1}$ 是回答这类问题的第一性公式。它是大模型训练四维并行（DP/TP/PP/EP）里 PP 维度的核心
权衡，几乎必问。

## 推导

一个 micro-batch 流过一段记 1 单位。GPipe 把前向排成对角线：第 `i` 个 micro-batch 在第 `j` 段占用时间槽
`i+j`。于是：
- 最早完工槽 = $(m-1)+(p-1)$，即 makespan $T = m+p-1$。
- 每段忙 $m$ 个槽（处理 $m$ 个 micro-batch），其余 $T-m=p-1$ 个槽是气泡。
- 利用率 $=m/T$，气泡率 $=(p-1)/T$。

对称 fwd/bwd（一前一后各占相同时间）时，分子分母同乘 2 再约掉，比值只与 $m,p$ 有关——这就是经典结论
「气泡只看 stage 数和 micro-batch 数」。

## 关键直觉

$$ \text{bubble}=\frac{p-1}{m+p-1}\ \Rightarrow\ \text{要压气泡就得 } m\gg p $$

- $m$ 固定、$p$ 越大 → 气泡越大（边际收益递减，这也是 PP 通常只在单机内做到几十 stage 的原因）。
- 想上更多 stage 又保持效率，就得把 global batch 切更多 micro-batch（受显存与有效 batch 限制）。

## 与其他题的关系

本题只算**气泡/时间**；显存维度见后续可能的 1F1B 题。通信维度：PP 只在相邻 stage 间传激活（点到点，
量很小），与 All-Reduce 类集合通信（`parallel-allreduce`、`distributed-zero-communication`）是不同的
瓶颈来源——这也是「PP 省的是层数显存、代价是气泡，TP/DP 代价是带宽」这一整体图景的一环。

## 实现与判题

闭式三行即可。`test_cases.py` 用**对角填充数忙碌格**的独立模拟做 oracle（不直接抄公式），并校验
`bubble+util==1`、$p=1$/$m=1$ 边界、$m$ 增大气泡单调下降。
