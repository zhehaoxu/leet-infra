# RL 基建：Rollout 数据跨 rank 重分片与 padding-free 打包

## 题目描述

在 verl / OpenRLHF 这类框架里，PPO 一步的数据要经过两次**重新划分**：生成侧（vLLM）按自己的分片把 $N$ 条 rollout 铺出来，训练侧的 $W$ 个数据并行 rank 却要求**按 token 数负载均衡**——否则抱着几条长尾序列的那个 rank 会把整步拖成 straggler（verl 的 `rebalance_batch`、DAPO 的 global partition 干的就是这件事）。均衡完还要在每个 rank 内部把变长序列做 **padding-free 打包**：靠 flash-attention 的 varlen 接口把多条序列首尾相接喂进一个微批，与「每条补齐到本 rank 最长条」的传统序列级 padding 相比，省下的 pad token 就是这一步的净收益。

本题不碰任何张量内容，只做**下标与整数记账**。

## 规则

输入三个量：`seq_lens`（int64 $(N,)$，每条 rollout 序列的 token 数，$\ge 1$，顺序即生成分片顺序）、`num_ranks`（$W$）、`micro_budget`（$M$，单个微批的 token 预算）。

1. **初始归属**：第 $i$ 条序列生成后属于 rank $\text{init}(i)=\lfloor i\cdot W/N\rfloor$（连续分块，不是轮询）。
2. **重分片**：处理顺序 = 序列按 $(\text{长度},\ \text{原下标})$ **升序**；依次把每条序列交给「当前真实 token 总数最少」的 rank（并列时取 rank 下标最小者）。得 `dest_rank`。
3. **微批打包**：每个 rank 内部沿第 2 步的顺序做贪心装箱——若当前包**非空**且加入本条会超过 $M$，就关掉当前包另起一个；若某条序列自身长度就已超过 $M$，它**独占一包**（包可以超预算，序列不可拆）。微批全局编号按 rank 升序、rank 内按出现顺序从 $0$ 连续递增，得 `micro_batch`。
4. **开销账**：`tokens_per_rank[r]` $=$ rank $r$ 的真实 token 和（padding-free 的代价）；`pad_saved[r]` $=$ 该 rank 的条数 $\times$ 该 rank 最长条 $-$ 真实 token 和，即改用序列级 padding 会多算的 token 数。`moved[i]` $=1$ 当 `dest_rank[i] != init(i)`，否则 $0$——它统计这次重排的跨卡搬运条数。

返回五元组（全部 `int64`）：`dest_rank` $(N,)$、`micro_batch` $(N,)$、`tokens_per_rank` $(W,)$、`pad_saved` $(W,)$、`moved` $(N,)$。

## 示例

`seq_lens=[30,10,50,20,40,60,15,25]`（$N=8$，总 token $=250$）、`num_ranks=3`、`micro_budget=70`：

```
均衡顺序(长度升序, 等长按原下标): 1(10) 6(15) 3(20) 7(25) 0(30) 4(40) 2(50) 5(60)
dest_rank       = [1, 0, 0, 2, 2, 1, 1, 0]
tokens_per_rank = [85, 105, 60]                 # 85+105+60 = 250

rank0 拿到 {1,7,2}: mb0=[10+25=35], mb1=[50]     # 35+50=85 > 70 → 另起一包
rank1 拿到 {6,0,5}: mb2=[15+30=45], mb3=[60]
rank2 拿到 {3,4}  : mb4=[20+40=60]
micro_batch       = [2, 0, 1, 4, 4, 3, 2, 0]     # 共 5 个微批，编号 0..4

序列级 padding 代价: rank0 3 条补齐到 50 → 150, rank1 → 180, rank2 → 80（合计 410）
pad_saved         = [65, 75, 20]                 # 410 − 250 = 160 个 pad token 被省掉

初始归属 init     = [0, 0, 0, 1, 1, 1, 2, 2]
moved             = [1, 0, 0, 1, 1, 0, 1, 1]     # 6 条需要跨卡搬运
```

## 约束

- CPU；$N\le 4096$，$1\le W\le 64$，$M\ge 1$，`seq_lens[i] ≥ 1`；五个输出均为 `int64`，判定用 `torch.equal`。
- 不变量：$\sum_r$ `tokens_per_rank[r]` $=\sum_i$ `seq_lens[i]`；`pad_saved[r] ≥ 0`（该 rank 内长度全相等时恰为 $0$）；`micro_batch` 用到的编号恰是连续集合 $\{0,\dots,B-1\}$；每个微批的 token 和 $\le M$，除非该批只含一条序列；空 rank 的 `tokens_per_rank`/`pad_saved` 为 $0$ 且不产生任何微批。
- 边界：$W=1$ 时全部序列落在 rank 0、`moved` 全 $0$；$W>N$ 时至多 $N$ 个 rank 拿到序列；$N=0$ 时 `dest_rank`/`micro_batch`/`moved` 为空张量、`tokens_per_rank`/`pad_saved` 为长度 $W$ 的全零；$M\ge$ 某 rank 的总 token 时该 rank 只有 1 个微批。
- 分工：`sequence-pack-pad` 做的是**一条流内部**的定长块打包，产出 token 级 `position_ids`/`segment_ids`；本题产出**跨 rank 归属、微批编号与两种 padding 的成本差**，不生成任何 token 级张量。损失/优势公式在 `to-*` 合集，通信量账在 `distributed-*` 合集。
