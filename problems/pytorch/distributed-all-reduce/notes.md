# 解题思路

## 面试官意图

All-Reduce 是数据并行的命脉：每张卡算自己那份 mini-batch 的梯度，然后必须**把梯度求和同步到所有卡**，
才能保证大家更新的是同一个模型。「DDP 一步里到底通信了什么、为什么是 All-Reduce」是训练岗必问题。

## 语义：归约 + 人人完整

All-Reduce 与 Reduce-Scatter 的唯一区别在**输出体积**：
- Reduce-Scatter：每 rank 只拿 $\frac1W$ 的归约结果；
- All-Reduce：每 rank 拿**完整**的归约结果（把上一步的 All-Gather 也做了）。

$$ T = \sum_{r} \texttt{inputs[}r\texttt{]},\quad \text{每个 rank 都得到 } T $$

## 恒等式：All-Reduce = All-Gather ∘ Reduce-Scatter

测试 `check_equals_gather_of_scatter` 把这条关系固化下来：先 reduce-scatter 让每卡持有全局和的一段，
再 all-gather 把所有段拼回，就得到每卡完整的和——正是 Ring All-Reduce 的两阶段。理解了这条，
`parallel-allreduce` 那道 $2\frac{W-1}{W}S$ 通信量就是从「RS 走一圈 + AG 走一圈」推出来的。

## 梯度平均 vs 求和

DDP 实际用的是**梯度平均**：NCCL 的 All-Reduce 求和后，框架再除以 `world_size`。本题按集合通信原语的
标准语义只做**求和**；要平均，在 $T$ 上统一除以 $W$ 即可（是纯后处理，不改变通信结构）。

## 常见错误

- 只算和但只返回一个张量（应返回长度 $W$、每项都是 $T$ 的列表）。
- 把求和写成求平均（本契约是求和）。
- 用了 `+=` 原地改 `inputs[0]`，污染输入。

## 判题口径

`test_cases.py` 用 Python 逐元素累加的独立参考核对，固化：所有 rank 相同且为完整和、
**All-Reduce == gather(scatter(...))** 的组合恒等、整型精确、单 rank 退化、不修改输入。
