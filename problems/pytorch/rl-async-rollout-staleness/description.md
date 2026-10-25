# RL 基建：异步 rollout 的准入、截断与陈旧度账

## 题目描述

同步 RLHF 的痛处很直白：一步里生成最长的请求（8k token 的 math 题）决定整步时间，其余卡只能等。于是 rollout 被拆成**异步服务**：生成端持续按 FCFS 接单产样本、放进 buffer；训练端每 `step_time` 出一个新策略版本；版本一更新，buffer 里的样本就开始"变旧"（off-policy）。AReaL 的做法是承认"样本一定旧"并把它变成可控量——给每条在飞请求一条**陈旧度截止线**，到点就提前截断交付（partial rollout），并用 $\mu$（消费时落后的版本数）作为上限去校正。verl / OpenRLHF 的 async / one-step-off-policy 路径是同一件事的另一个参数化。

本题把这套路抽成一张**纯整数事件表**：不生成任何 token、不需要策略模型，只按给定数据（每条请求的生成时长与到达时刻）模拟准入、截断、成批，然后算出陈旧度、占用时间线与训练器空转时长。

## 口径

约定**版本网格**：版本 $v$ 在时刻 $v\cdot\text{step\_time}$ 发布，在半开区间 $[v\cdot\text{step\_time},(v+1)\cdot\text{step\_time})$ 内生效。训练器即使 buffer 空也照常推进网格（空转）——`stall` 量的就是这个。所有时间都是整数单位（1 单位 = 生成 1 token）。

## 规则

输入六个量：`gen_lens` $(N,)$ int64（每条请求需要的生成时长，$\ge 1$）、`arrivals` $(N,)$ int64（到达时刻，**不要求有序**）、标量 `capacity` $=C\ge1$（并发槽数，槽之间等价）、标量 `step_time` $\ge1$、标量 `batch_size` $=B\ge1$、标量 `max_staleness` $=S\ge1$。

1. **准入（FCFS）**：排队顺序按 $(\text{arrivals},\ \text{请求号})$ 字典序。请求按此顺序逐个开工，开工时刻
   $$\text{start}_i=\max(\text{arrivals}_i,\ \text{最早空出槽的时刻})$$
   槽的空闲时刻初值 $0$；把请求分给"最早空出的那个槽"（并列取槽号最小，因槽等价而不影响任何结果），该槽的下一次空闲时刻改为这条请求的 `finish`。
2. **partial rollout 截止线**：入场版本 $v0_i=\lfloor \text{start}_i/\text{step\_time}\rfloor$，
   $$\text{deadline}_i=(v0_i+S)\cdot\text{step\_time},\quad
     \text{finish}_i=\min(\text{start}_i+\text{gen\_len}_i,\ \text{deadline}_i)$$
   `truncated[i]` $=1$ 当且仅当 $\text{finish}_i<\text{start}_i+\text{gen\_len}_i$——**刚好卡在截止线上完成不算截断**。$S\ge1$ 保证 $\text{finish}_i\ge \text{start}_i+1$（因为 $\text{start}_i-v0_i\cdot\text{step\_time}\le \text{step\_time}-1$），不会出现零时长请求。
3. **按完成顺序成批**：把请求按 $(\text{finish},\ \text{请求号})$ 字典序排队，名次为 $q$ 的请求进 `batch_id` $=\lfloor q/B\rfloor$ 这一批（末批可以不满 $B$ 条）。整批**共用**一个消费版本：$c_k=\lfloor \text{ready}_k/\text{step\_time}\rfloor$，$\text{ready}_k$ 是批内最晚的 `finish`（短板效应的来源）。
4. **陈旧度**：`mu[i]` $=c_{\text{batch\_id}[i]}-v0_i$，即"这条样本开始生成时的策略"与"它被消费时的策略"之间隔了几个版本。恒有 $\text{mu}\ge0$（$\text{ready}\ge\text{finish}\ge\text{start}$）。
5. **占用时间线**：`occupancy[tau]` $=\#\{i:\text{start}_i\le \text{tau}<\text{finish}_i\}$，$\text{tau}=0,1,\dots,\max_i\text{finish}_i-1$，故其长度就是最晚完成时刻（不含终点）。
6. **空转**：`stall[k]` $=\max\!\big(0,\ (c_k-k)\cdot\text{step\_time}\big)$，长度 $\lceil N/B\rceil$。理想流水线下第 $k$ 批正好在版本 $k$ 被消费；生成跟不上就出现正的 stall，供过于求则被 clamp 到 $0$（版本网格不会倒着走）。

返回五元组（全 int64）：`mu` $(N,)$、`batch_id` $(N,)$、`truncated` $(N,)$、`occupancy` $(\max\text{finish},)$、`stall` $(\lceil N/B\rceil,)$。$N=0$ 时五个都是长度 $0$ 的空张量。

## 示例

$N=8$、$C=2$、$\text{step\_time}=4$、$B=3$、$S=2$，`gen_lens=[6,7,9,5,4,8,3,10]`、`arrivals=[0,0,1,2,5,6,10,11]`。中间量表（按题面规则逐步推出，便于自查）：

```
 i  到达 生成 start v0 deadline finish 截断 | 批号 ready c  mu
 0   0   6     0   0     8      6     0 |  0   12 3   3
 1   0   7     0   0     8      7     0 |  0   12 3   3
 2   1   9     6   1    12     12     1 |  0   12 3   2
 3   2   5     7   1    12     12     0 |  1   19 4   3
 4   5   4    12   3    20     16     0 |  1   19 4   1
 5   6   8    12   3    20     20     0 |  2   24 6   3
 6  10   3    16   4    24     19     0 |  1   19 4   0
 7  11  10    19   4    24     24     1 |  2   24 6   2
```

```
mu        = [3, 3, 2, 3, 1, 3, 0, 2]
batch_id  = [0, 0, 0, 1, 1, 2, 1, 2]
truncated = [0, 0, 1, 0, 0, 0, 0, 1]
occupancy = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1]   # 长度 24 = 最晚完成时刻
stall     = [12, 12, 16]        # 消费版本 [3,4,6] vs 理想 [0,1,2]
```

几点读法：请求 3 本该在 $7+5=12$ 完成、截止线也是 $12$，**相等 ⇒ 不算截断**；请求 5 与 7 同进批 2，被 7 拖到 $\text{ready}=24$，于是本来只落后 1 个版本的 5 也变成 $\mu=3$；被截断的两条都满足 $\mu\ge S=2$。

同一组数据只改 $S$（其余不变）：

```
S=1  → truncated=[1,1,1,1,0,1,0,1]  mu=[2,2,1,2,1,1,1,1]  batch_id=[0,0,0,1,1,1,2,2]
       stall=[8,8,8]   occupancy 长度 16 = [2]*15+[1]
S=10 → truncated 全 0  mu=[3,3,3,2,1,4,0,3]  batch_id=[0,0,1,0,1,2,1,2]
       stall=[12,12,20]  occupancy 长度 29 = [2]*23+[1]*6
```

$S$ 一改，截断一改，**完成顺序就变了，成批归属跟着变**（请求 2 从批 0 挪到批 1），连空转时长也变了——这就是"截断换新鲜度"的代价账。

## 约束

- CPU；$0\le N\le 256$、$1\le C\le 64$、$1\le \text{step\_time}\le 64$、$1\le B\le 256$、$1\le S\le 32$；`gen_lens` $\ge1$、`arrivals\ge0$；全部整数运算，无浮点误差问题。
- 判定：五个输出全部用 `torch.equal`（int64）。
- 退化不变量：$C\ge N$ 且 $S$ 足够大 $\Rightarrow$ `start` 等于到达、`occupancy` 峰值 $=N$；$C=1\Rightarrow$ `occupancy` 全 $1$（串行，长度 $=\sum$ 时长）；$B\ge N\Rightarrow$ 只有批 0、`stall` 长度 $1$；$S$ 极大 $\Rightarrow$ `truncated` 全 $0$；供给充裕 $\Rightarrow$ `stall` 全 $0$；被截断的样本必有 $\text{mu}\ge S$。
- 分工：`inf-continuous-batching-scheduler` 只算"FCFS 完成后每条请求几点结束"，没有版本轴、没有截断、没有消费侧成批；本题复用它的准入内核，新增的是**截止线反馈进时间线**、**完成顺序成批**与**陈旧度/空转**。$\mu$ 之后怎么进损失（off-policy 校正）属 `to-ppo-clipped-surrogate` / `to-kl-controller`，本题只把 $\mu$ 算准。
