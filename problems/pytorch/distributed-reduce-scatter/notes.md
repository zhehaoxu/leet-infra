# 解题思路

## 面试官意图

「All-Reduce = Reduce-Scatter + All-Gather」是分布式通信最常被引用的一句话。要理解它，必须先各自
吃透两个半程。Reduce-Scatter 的输出**每个 rank 都不一样**（各拿一段归约结果），这点和 All-Gather
「人人相同」正好相反——面试最爱在这里设陷阱。

## 两步拆解

1. **Reduce**：把 $W$ 份等长数据逐元素归约（求和）成一份长度 $L$ 的结果 $T$。
2. **Scatter**：把 $T$ 切成 $W$ 段，第 `i` 段发给 rank `i`。

所以 rank `i` 的输出 $=T[i\cdot\text{chunk}:(i+1)\cdot\text{chunk}]$，`chunk = L/W`。返回的是**每 rank
结果不同**的列表。

## 与 Ring All-Reduce 的关系

Ring All-Reduce（见 `parallel-allreduce` 的通信量分析）就是：
- 阶段一 Reduce-Scatter：转 $W-1$ 圈后，每卡恰好持有「全局和」的 $\frac1W$；
- 阶段二 All-Gather：再转 $W-1$ 圈，把各自的 $\frac1W$ 拼给所有人 → 每卡拿到完整全局和。

两步通信量各 $(W-1)/W\cdot S$，合起来就是 $2\frac{W-1}{W}S$。

## 真实场景：ZeRO

ZeRO-1/2 的通信正是 Reduce-Scatter：优化器状态/梯度被切成 $W$ 份，每卡只负责自己那 $\frac1W$ 的
更新，天然把优化器显存摊薄到 $1/W$（对应 `training-state-memory` 里的分区公式）。

## 常见错误

- 每 rank 都返回完整 $T$（那是 All-Reduce，不是 Reduce-Scatter）。
- 切片顺序错（把第 `i` 段给了 rank `W-1-i`）。
- 用 `reshape` 前没保证能被 $W$ 整除（本契约已保证整除，但仍应显式取 chunk）。

## 判题口径

`test_cases.py` 用手写逐元素求和 + 切片的独立参考核对，并固化不变量：**各 rank 分片 `cat` 起来
== 完整逐元素和**、rank 顺序正确、不修改输入。
