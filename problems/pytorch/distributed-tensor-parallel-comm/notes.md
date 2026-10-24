# 解题思路

## 面试官意图

「TP 和 DP 的通信有什么本质不同？」「为什么 TP 度数一般不超过单机 8 卡、而 DP 能拉到几千卡？」——关键就在
**同步对象**：DP 同步梯度（$\propto$ 参数量、每步一次），TP 同步激活（$\propto$ batch·seq·hidden、**每层每次
前向反向都要**）。算清 TP 每层通信字节，就懂了 TP 为什么对带宽如此敏感、为什么离不开 NVLink。

## 每层 4 次 all-reduce 的来源

Megatron 把一个 transformer 层切成：
- 注意力：`QKV` 列并行 → `out_proj` 行并行 → 行并行输出是部分和，需 **1 次 all-reduce**（前向）。
- MLP：`fc1` 列并行 → `fc2` 行并行 → 同样 **1 次 all-reduce**（前向）。
- 反向对称再来 2 次。

于是**每层 4 次 all-reduce**，每次消息是一张激活 `act_bytes = batch·seq·hidden·dtype`。$W$ 卡环上一次
all-reduce 每 GPU 收发 $2c\cdot\text{act}$（$c=\frac{W-1}{W}$），合起来 $\text{per\_layer}=8c\cdot\text{act}$。

## 为什么 TP 饱和快

$W$ 变大时 $c\to1$，$\text{per\_layer}\to 8\cdot\text{act}$：通信量**几乎不随 $W$ 降**（每卡都要搬约整张激活），
而计算被 $W$ 卡摊薄，于是**通信/计算比随 $W$ 线性恶化**——这就是 TP 度数一小就撞带宽墙、只在 NVLink 机内做
的原因。对比 DP：梯度 all-reduce 每步一次、量 $\propto$ 参数量，可被整步计算充分摊薄，故能上几千卡。

## 实现要点

先算 `act_bytes`（与 world 无关，`check_act_bytes_independent_of_world` 卡这点），乘环系数与 4 次，再乘层数。
$W=1$ 时 $c=0$、后两项归零。判题独立参考把 4 次 all-reduce **逐次累加**得到每层量（不直接写 $8c$ 闭式）。

## 相关题

`parallel-allreduce` / `distributed-zero-communication`（同一环系数 $2c$，但那是梯度）、`distributed-gather` 等
原语。加上 DP/PP/EP，本题补上四维并行里的 **TP** 通信账；Sequence Parallel 会用 all-gather+reduce-scatter
替代部分 all-reduce 进一步省显存/带宽，属进阶扩展（题面刻意不建模）。
