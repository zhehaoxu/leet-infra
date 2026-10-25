# 详解：Rollout 数据跨 rank 重分片与 padding-free 打包

## 出处
- verl 的 PPO/DAPO 训练器在把生成结果交给 DP 训练侧之前会做 `rebalance_batch`（DAPO recipe 里配套 `get_global_partition_idx_and_max_response_length`）：按序列长度排序后切**全局分片**，让每个 rank 的 token 负载尽量相等，并记录每片的**最长条**（它决定该片 padding 的宽度）。
- OpenRLHF / verl 的 padding-free 微批切分见 `verl/utils/seqlen_balancing.py`（`rearrange_micro_batches` 一系），本质是「在 token 预算下装箱 + 产出 `cu_seqlens` 边界」，配合 flash-attention 的 varlen 接口把多条序列拼成一条流。
- 「序列级 padding 的浪费 = 条数 × 最长条」这一口径来自 DP 同步训练：一个 rank 的一批变长序列必须补成同宽才能进同一个 forward。

## 核心约定（写偏一条就全错）
1. **均衡顺序**：处理顺序是 $(\text{长度},\text{原下标})$ **升序**。升序（而不是 LPT 的降序）配「最短队列优先」，把最大的几条留在最后分发，得到的负载极差被**最大单条**控制；等长按原下标升序是必要的 tie-break，否则同一份数据两次跑会给出不同归属。
2. **贪心目标**：每次选「当前**真实 token 总数**最少」的 rank，并列取最小 rank 下标。注意键是 token 数而不是序列条数——RL rollout 的耗时由 token 数决定，按条数均衡在有长尾时几乎必然失效（见 `check_rebalancing_reduces_skew`：15 条里 3 条占 350 token）。
3. **微批编号**：rank 升序、rank 内按装箱出现顺序，全局从 $0$ **连续**编号。真实框架里每个 rank 本地从 0 编号（微批号只在本 rank 有意义），本题为了可判等把它拉平成全局编号；空 rank 不消耗编号。
4. **单条超预算不拆**：装箱遇到「当前包非空且加入会超预算」才关包，所以任何一包 $\le M$，**除非**它只含一条长度本身 $>M$ 的序列。padding-free 的边界是序列整体，不能把一条序列切成两半（`cu_seqlens` 会伪造出一条不存在的序列终点）。

## pad_saved 的口径与常见写错
`pad_saved[r] = count_r × max_len_r − Σ len`。三种典型错法：
- 把 `max_len` 取成**全局**最长（跨 rank）→ 高估浪费；
- 把 `count` 取成 $N$ 而不是该 rank 的条数；
- 忘记空 rank → `count=0` 时该值为 $0$（若按「$0\times$ 全局最长」写法也可能得 0，但只要多一条特殊分支就容易漏）。
正确实现里 `pad_saved[r] ≥ 0` 恒成立，且当该 rank 内长度全相等时**恰为 $0$**：这是最强的自检（`check_equal_lengths_round_robin`）。它省的是**无效算力**（pad token 也要过前向/反向），不是显存；显存账在 `rl-colocate-memory-timeline` 与 `training-state-memory`。

## 边界约定
- **初始归属是连续分块** $\lfloor iW/N\rfloor$，不是轮询 $i\bmod W$。写成轮询会让 `moved` 大面积变 0，掩盖真实搬运量。
- $W>N$ 时最多 $N$ 个 rank 有活，空 rank 的 `tokens_per_rank`/`pad_saved` 是 $0$（不是缺项）。
- $N=0$ 时前三个/后两个输出长度不同：序列级输出是空张量，rank 级输出是长度 $W$ 的零向量。

## 与其他题族的分工
- `sequence-pack-pad`：在**一条流内部**把序列打包成定长块，产出 token 级的 `input_ids/position_ids/segment_ids/attention_mask`；本题**不生成任何 token 级张量**，只产出「哪条序列归哪个 rank、第几个微批、两种 padding 的代价差、几条要搬运」。教学点在**跨 rank 重分布**，padding 只是成本口径。
- `distributed-zero-*` / `distributed-pipeline-*`：算模型状态的切分归属与通信/气泡；这里的「分片」是**数据**的分片，搬运条数是唯一的通信代理指标（没有换算成字节，字节账依赖具体模型，归 `distributed-*`）。
- `rl-colocate-memory-timeline`：管显存与阶段时间线；本题管 token 负载与装箱。
- `to-*`：损失/优势公式；本题连 reward 都不看，只看长度。

## 复杂度
排序 $O(N\log N)$ + 每条序列扫一遍 rank 求最短队列 $O(NW)$，$N\le 4096, W\le 64$ 下无需堆优化。oracle 把负载**每次现算求和**、装箱用指针 while、初始归属用 ceil 边界成员计数、浪费用逐条补差求和——四条路径都与参考解结构不同，可用来交叉验证。
