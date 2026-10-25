# 详解：异步 rollout 的准入、截断与陈旧度账

## 出处
- **异步 / partial-rollout 的 RLHF 服务**：AReaL（Infinite Light + 清华，arXiv 编号待回源）把"样本一定 off-policy"当作一等公民——训练端持续更新权重，生成端不停机接单，给每条在飞请求挂一个**陈旧度上限**，到点就打断生成、交付半成品（partial rollout）。verl（ByteDance，Seed）的 async / one-step-off-policy 路径、以及 OpenRLHF 的异步 vLLM 生成服务，是同一件事的另一种参数化：它们的 `max staleness` / `partial rollout` 开关（具体字段名待回源）就是本题里的 $S$。本题**不依赖任何这些库**（判定环境里也没有），只把这套调度抽成一张纯整数事件表。
- **off-policyness 的度量约定**：$\mu=\text{消费版本 }c-\text{入场版本 }v0$，即"这条样本**开工时**生效的策略"与"它**被吃进训练**时生效的策略"之间隔了几次更新。有的论文把它定义成"生成期间发生了多少次权重同步"，与本题的 $c-v0$ 只差一个"是否含当前步"的下标习惯；这里统一按题面：**入场版本用开工时刻 $v0=\lfloor start/st\rfloor$ 定位，消费版本用整批 ready 时刻 $c=\lfloor ready/st\rfloor$ 定位**。
- 本题与真实框架的**抽象差**：真实系统里 $st$ 是一步训练的墙钟、partial rollout 会跨多次权重同步续跑；本题把这些压成单张时间轴上的整数网格，token 数 = 时间单位，无随机、无模型前向。

## 完整推导与下标约定
1. **槽位递推（准入内核）**：维护 $C$ 个等价槽的下一次空闲时刻 `release[0..C-1]`，初值全 $0$。按 $(\text{arrivals},i)$ 升序逐条处理请求 $i$：
   - 选 $j^\star=\arg\min_j \text{release}[j]$（**并列取最小槽号**），
   - $\text{start}_i=\max(\text{arrivals}_i,\ \text{release}[j^\star])$，
   - 算出 $\text{finish}_i$（见下）后令 $\text{release}[j^\star]\leftarrow \text{finish}_i$。
   这正是"最早空出的槽接下一条"，与 `inf-continuous-batching-scheduler` 的 continuous batching 同源；槽等价 ⇒ 并列取哪个槽不影响任何**输出**（只影响中间 `release` 的槽号标签，而输出里没有槽号）。
2. **版本网格**：版本 $v$ 只在半开区间 $[v\cdot st,(v+1)\cdot st)$ 内生效。$\lfloor t/st\rfloor$ 就是"$t$ 落在哪个格子里"，是本题所有"版本"下标的唯一来源。网格是**发布**驱动的，与 buffer 是否有货无关：训练器没数据也照常推进（空转）。
3. **截止线为什么钉在入场版本 $v0$ 而非当前版本**：partial rollout 的语义是"这条样本最多允许跨 $S$ 个版本再交付"。它在 $v0$ 格子开工，于是硬截止是 $v0$ 往后数 $S$ 格的左端点 $\text{deadline}=(v0+S)\cdot st$。若把 deadline 钉在"当前版本"上就会变成滑动、可无限续生，staleness 失去上界——所以截止线对**入场时刻**取定后不再变。由 $S\ge1$ 与 $0\le start-v0\cdot st\le st-1$ 得 $\text{deadline}\ge start+1$，配合 $\text{finish}=\min(start+\text{gen\_len},\text{deadline})$ 且 $\text{gen\_len}\ge1$，恒有 $\text{finish}\ge start+1$（无零时长请求）。
4. **完成与截断**：$\text{finish}_i=\min(\text{start}_i+\text{gen\_len}_i,\ \text{deadline}_i)$；$\text{truncated}_i=[\text{finish}_i<\text{start}_i+\text{gen\_len}_i]$。注意是**严格小于**才算截断——恰好等于 deadline（$\text{gen\_len}$ 正好填满剩余窗口）不算截断。
5. **按完成顺序成批（消费侧）**：把请求按 $(\text{finish},i)$ 升序排队，名次 $q$ 的请求进 $\text{batch\_id}=\lfloor q/B\rfloor$。整批的可用时刻取**批内最晚的 finish**（短板效应：$\text{ready}_k=\max_{i\in k}\text{finish}_i$），于是 $\text{consume}_k=\lfloor \text{ready}_k/st\rfloor$ 被批内最长的那条拖后。$c$ 对整批共用，故短样本会替长样本"陪绑"变旧。
6. **陈旧度与空转**：$\text{mu}_i=\text{consume}_{\text{batch\_id}[i]}-v0_i$。空转 $\text{stall}_k=\max(0,(\text{consume}_k-k)\cdot st)$——理想流水线下第 $k$ 批应在版本 $k$ 被吃（消费版本 $=k$），落后就 $\text{consume}_k>k$ 产生正空转，供过于求被 clamp 到 $0$（格子不会倒着走）。$(c_k-k)$ 这个"消费版本 − 理想版本"的差，量的是**生成跟不上训练**的步数。

## 核心与易错点
- **并列槽位取最小槽号**：`release` 里多个槽同时最早空闲时选槽号最小者。因槽等价，这条只锁死中间态、不影响最终五元组；但写成"随便取一个"会让逐时刻推进的 oracle 与逐条推进的参考解在**中间量表**上对不齐（调试时最容易误判成 bug）。
- **刚好卡在 deadline 上完成不算截断**：判定是 $\text{finish}<\text{start}+\text{gen\_len}$ 的严格不等号。把 $=$ 也算截断是头号错误（示例里请求 3 的 $7+5=12=\text{deadline}$ 就是这条边界，`truncated=0`）。反向的"差 1 个单位就必截断"见 `check_partial_rollout_trims_long_tail` 的 `[9]` 用例。
- **occupancy 用半开区间 $[\text{start},\text{finish})$**：$\text{occupancy}[\tau]=\#\{i:\text{start}_i\le\tau<\text{finish}_i\}$，长度 $=\max_i\text{finish}_i$（不含终点）。写成闭区间会在每条请求的最后 1 格多数一次占用、且 $\sum\text{occupancy}$ 不再等于总忙时。自检：`check_invariants` 里 `occ.sum() == Σ(finish-start)`。
- **stall 长度 $\lceil N/B\rceil$**：批数是向上取整，末批可不满 $B$。写成 $\lfloor N/B\rfloor$ 会漏掉尾巴那一整批。`check_single_batch_when_batch_size_ge_n` 与 `check_matches_oracle_batch_size_sweep` 专门钉这条。
- **$N=0$ 的空张量约定**：五个输出**都是**长度 $0$ 的 int64 空张量（`occupancy`/`stall` 不是长度 1 的零向量）。且必须 `.to(torch.long)`——空张量默认 dtype 是 float，判定用 `torch.equal` 会因 dtype 不符挂掉。
- **$S$ 一改全盘皆动**：截断改写 finish ⇒ 完成顺序变 ⇒ 名次切批变 ⇒ $\mu$/stall/occupancy 全变（示例 $S=2$ 里请求 2 在批 0，$S=10$ 里请求 2 挪到批 1）。所以 $S$ 不只是截断阈值，它是**反馈进整条时间线**的旋钮。
- **被截断必 $\text{mu}\ge S$**：截断的样本交付在 $\text{deadline}=(v0+S)st$ 或更晚的批 ready 上，故 $\text{consume}\ge v0+S$。（不等号方向别写反。）

## 与邻近题的区别
- **vs `inf-continuous-batching-scheduler`**：它的 `solve(requests, max_batch_size)` 只把 $(N,2)$ 的 `[arrival_step, gen_len]` 按 FCFS continuous batching 排一遍，**唯一输出是每条请求的完成时刻** `finish (N,)`。它是**推理服务**口径：固定并发上限、请求生完即走、**没有策略版本轴**、没有截止线、没有消费侧成批。本题**复用它的准入内核**（最早空闲槽 + `start=max(arrival, free)`），但新增三件事：① finish 被**入场版本衍生的 deadline 削平**并反馈回时间线（partial rollout），② 按 $(\text{finish},i)$ 名次**成批 + 整批共用消费版本**，③ 输出**陈旧度 mu、占用时间线 occupancy、空转 stall**。另外本题的 `arrivals` 显式不要求有序、靠请求号破并列，`capacity` 是等价并发槽而非"最大 running batch"。
- **vs `perf-kernel-launch-timeline`**：那条也是"忙/闲账"，但对象是**单条串行流上的 kernel 发射**（$r_i=(i+1)L$、$s_i=\max(r_i,e_{i-1})$、busy_ratio、launch_bound_count），算的是 CPU 发射跟不上 GPU 的空转，**无并发槽、无版本网格、无 staleness**。本题的 idle（stall）是**训练器因生成端跟不上**而空转，语义完全不同。
- **vs `to-*` 系列的 PPO/GRPO 目标函数**（`to-ppo-clipped-surrogate` 截断代理目标、`to-grpo-advantage` 组内标准化优势、`to-kl-controller` 自适应 KL 系数）：它们是**已采到样本之后**的损失/优势/系数公式，$\mu$ 在那些题里是作为 off-policy 校正的**输入**。本题站在**数据供给侧**：只把 $\mu$、成批、截断算准，**一个损失数值都不算**。
- **vs 兄弟题 `rl-colocate-memory-timeline`**：那条管多模型（actor/critic/ref/reward）colocate 的**显存峰值 / offload 传输 / 阶段空闲时间线**；本题管**单个生成池**的 rollout 时间与 staleness，二者一个是显存账、一个是生成调度账。**vs `rl-rollout-shard-repack`**：那条把已生成的序列**跨 DP rank 重分片 + padding-free 装箱**、算 token 负载与搬运条数；本题根本不碰"序列归哪个 rank"，只算"什么时候交付、交付时旧了几版"。
