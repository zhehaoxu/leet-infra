# 解题说明 · PD 分离的放置账

## 出处

- **Mooncake**（arXiv:2407.00079，Moonshot AI）：KVCache-centric 的分离式架构——prefill 池与 decode 池分开，KV 从 prefill 节点经 RDMA 搬到 decode 节点，并把 KV cache 当成一级资源做放置与调度。本题的「prefill 完成拍 → KV 搬运 → decode 侧就位」链路取自这一形态。
- **DistServe**（arXiv:2401.09670）：把 prefill 与 decode 解耦后，用 TTFT/TPOT 两个 SLO 下的 goodput 评价放置；本课把其中的「KV 传输与显存准入决定请求能否被接纳」抽象成确定性账本。
- 环形交换的链路份额口径与本仓库 `distributed-sequence-parallel` 一致：C 个 rank 的 all-gather，每条链路只承担 `(C-1)/C` 的份额。这里把节点内 TP 交换也记在同一条链路上，属**简化假设**。

## 核心与易错点

**prefill 取整在累计量上做。** `ceil(cum/tps)` 与「每请求各自 ceil 再累加」在 tps 不能整除时结果不同（示例：3 个长度 3 的请求、tps=2，累计口径是 2/3/5 拍，逐请求口径是 2/4/6 拍）。写成后一种，TTFT 会随请求数线性膨胀。

**被丢弃的请求仍然吃掉了 prefill。** 准入失败的请求不占显存、不占链路，但它在队列里排队的那段时间是真实发生的；后续请求的 `prefill_done` 必须继续按包含它的累计量算。把丢弃请求从累计量里剔掉，会让后面所有请求的 TTFT 集体提前。

**环形份额是每条链路的份额，不是总流量。** `moved = kv + ceil(kv*(C-1)/C)`：tp=1 时没有这一项；tp=2 时等于多搬一半；tp=4 时多搬 3/4。别写成 `kv*C`（那是广播的总流量）或 `kv*(C-1)/C`（少了 handoff 本身）。

**容量按峰值算。** `kv = (prompt + gen) * bytes_per_token`，生成阶段 KV 还会继续长；只按 prompt 算容量会接纳过多请求。准入判据是 `resident + kv <= caps`，等号算装得下。

**放置按「预计完成拍」比，不是按「谁最闲」。** 空闲更早但链路慢的副本，可能输给空闲稍晚但带宽大的副本；这条在 `check_earliest_finish_wins` 里专门构造过。平手才回落到下标小的。

**预估要在提交前算。** 四个候选副本的 `start` 都必须用**本请求提交前**的 `link_free`；若边算边更新，先算到的副本会污染后算的副本的估计值。

**TTFT 的口径是 KV 就位**，不含 decode 首步计算；题面统一按这一拍比较，别再额外加一拍常数。

## 与其他题的分工

- `inf-cache-aware-routing`：按 prefix cache 命中选实例，考的是命中与路由；本题没有 cache，只有搬运与准入。
- `inf-kv-residency-billing`：块·步积分与计费；`inf-paged-cow-block-share`：块级 COW；`inf-kv-block-admission`：块池准入。三者都是单实例块管理，不涉及跨节点 KV 搬运。
- `inf-preempt-swap-recompute`：显存不足时抢占在跑请求并比较 swap/recompute；本题是**新请求**的放置与丢弃，不做抢占。
- `distributed-sequence-parallel`：只给 `2*f*A*(C-1)/C` 的一条显存-通信公式；本题复用了它的环系数，但落到多副本、多请求的排队与准入上。
