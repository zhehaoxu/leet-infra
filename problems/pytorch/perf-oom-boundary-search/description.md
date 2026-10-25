# OOM 边界搜索：峰值显存反解 micro batch 与序列长度

## 题目描述

调 batch 时工程师面对的是两个反问题：**序列长度已经定了，micro batch 还能开多大？micro batch 定了，序列最长
能拉到多少？** 峰值显存里，参数、梯度、优化器状态是一块与 batch 无关的**静态占用**，激活则是
「每 micro batch 每 token 的线性项」加上「注意力分数带来的 $O(s^2)$ 项」——前者对 $m$ 线性、后者对 $s$ 是二次，
所以两条边界的性质完全不同：一个是整除 floor，另一个是二次方程正根，**取整方向差一点就是 OOM 与「刚好放下」
的区别**。算出边界之后，还要回答「这个 global batch 需要几个 rank、凑不凑得齐」。

`transformer-mem` / `training-state-memory` 给的是**字节数本身**（不随 batch 反解），
`distributed-zero-shard-ownership` 管分片归属，`kv-cache-memory` 管推理 KV；本题只管**给定字节预算后
向上/向下搜索最大可行配置**，以及数据并行度必须为整数带来的整除判定。

## 规则

- **可用额度** $\text{usable}=\texttt{hbm\_bytes}-\texttt{reserve\_bytes}$：预留是分配器碎片/驱动占用的额度，
  **不进峰值**；`cuda_overhead_bytes` 则**进峰值**。$\text{usable}$ 允许为负。
- **静态占用** $\text{static}=\text{param}+\text{grad}+\text{optim}$（ZeRO-0：每个 rank 都存全量）。
  默认题目用例按 `training-state-memory` 的 16 字节/参数拆开给（fp16 参数 2 + fp16 梯度 2 + fp32 master/m/v 12）。
- **峰值** $\text{peak}(m,s)=\text{cuda\_overhead}+\text{static}+m\cdot s\cdot\texttt{act\_per\_token}+m\cdot s^2\cdot\texttt{act\_per\_token\_sq}$（二次项**也乘** $m$，因为每条序列都有一份自己的 $s^2$ 分数矩阵）。
- **可行性**用**闭区间**：$\text{fits}\iff \text{peak}\le \text{usable}$，刚好用满算可行；
  $\text{headroom}=\text{usable}-\text{peak}$，**可为负、不 clamp**。
- **两条边界**：固定 $s$ 求最大整数 $m\ge1$、固定 $m$ 求最大整数 $s\ge1$；连 $1$ 都放不下就记 $0$。
  $m$ 那条是线性的，直接 $\lfloor \text{budget}/(s\cdot ap+s^2\cdot apq)\rfloor$；
  $s$ 那条是 $m\,apq\,s^2+m\,ap\,s\le \text{budget}$（$\text{budget}=\text{usable}-\text{cuda\_overhead}-\text{static}$），
  取二次方程正根的 floor 再按判据校正；$\texttt{act\_per\_token\_sq}=0$ 时退化为线性 floor。
- **数据并行度** $\text{world}=\lceil G/(A\cdot m)\rceil$ 用**给定的** `micro_batch`（不是 $\text{max\_micro\_batch}$）；
  $\text{global\_batch\_fits}=\text{fits}\wedge(\text{world}\cdot A\cdot m=G)$——rank 数必须是整数，
  最后一个小 micro batch 不许半空，所以 $G$ 不被 $A\cdot m$ 整除时为假。
- 输入含义、每条公式与返回类型口径全部写在 `starter.py` 的 docstring 里，以它为准。

## 示例

```
输入：hbm=80e9, reserve=2e9, cuda_overhead=1.5e9, param=2.6e9, grad=2.6e9, optim=15.6e9,
      act_per_token=1.5e6, act_per_token_sq=128, micro_batch=4, seq=2048, global_batch=512, accum=8
  usable = 78e9   static = 20.8e9   budget = 78e9-1.5e9-20.8e9 = 55.7e9
  peak(4,2048) = 1.5e9+20.8e9+4*2048*1.5e6+4*2048^2*128 = 36,735,483,648（线性 12.288GB + 二次 2.147GB）
  headroom = 41,264,516,352  feasible=True
  max_micro_batch = floor(55.7e9 / (2048*1.5e6+2048^2*128)) = floor(55.7e9/3,608,870,912) = 15
  max_seq：512 s^2 + 6e6 s <= 55.7e9 的正根 floor = 6103
  world = ceil(512/(8*4)) = 16，16*8*4 = 512 -> global_batch_fits=True
输出：(20800000000, 36735483648, 41264516352, True, 15, 6103, 16, True)

输入：同上，但 micro_batch=8, seq=8192
  peak(8,8192) = 1.5e9+20.8e9+8*8192*1.5e6+8*8192^2*128 = 189,323,476,736 > 78e9 -> OOM
  headroom = -111,323,476,736   feasible=False
  max_micro_batch = floor(55.7e9/20,877,934,592) = 2   max_seq（m=8）= 3560
  world = ceil(512/(8*8)) = 8，8*8*8 = 512 整除，但显存不可行 -> global_batch_fits=False
输出：(20800000000, 189323476736, -111323476736, False, 2, 3560, 8, False)

输入：hbm=20000, reserve=cuda_overhead=param=grad=optim=0, act_per_token=1000,
      act_per_token_sq=0, micro_batch=2, seq=10, global_batch=40, accum=4
  peak = 2*10*1000 = 20000 == usable -> 闭区间判可行，headroom=0
  max_micro_batch = 20000//(10*1000) = 2   max_seq = 20000//(2*1000) = 10
  world = ceil(40/(4*2)) = 5，5*4*2 = 40 -> True
输出：(0, 20000, 0, True, 2, 10, 5, True)
（把 hbm 改成 19999：feasible=False、headroom=-1、两条边界各掉一级变成 1 和 9）
```

## 约束

- `hbm_bytes` $>0$；`reserve_bytes`、`cuda_overhead_bytes`、`param_bytes`、`grad_bytes`、`optim_bytes`
  $\ge 0$；`act_per_token` $\ge 1$；`act_per_token_sq` $\ge 0$；`micro_batch`、`seq`、`global_batch`、
  `accum_steps` $\ge 1$
- 返回八元组 `(static_bytes, peak_bytes, headroom_bytes, feasible, max_micro_batch, max_seq,
  world_size_needed, global_batch_fits)`：第 1、2、3、5、6、7 项是 int（精确相等），第 4、8 项是 bool
- 全部按**整数字节**计算，不涉及浮点比较；`act_per_token >= 1` 保证两条边界一定有界，
  因此除了「连 1 都放不下记 0」之外没有其它哨兵值
