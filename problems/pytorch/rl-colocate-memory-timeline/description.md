# RL 基建：多模型 colocate 的显存与阶段时间线账

## 题目描述

RLHF 一步里同时住着好几个模型：actor 要训练、critic 要训练、reference 与 reward 只推理。把它们**放进同一张卡**（colocate）就必然要在阶段之间搬进搬出：生成时把 ref/critic 的权重 offload 到 CPU、把 KV cache 和激活留给 actor；轮到 `ref_logp` 再把 ref 换回来。verl / OpenRLHF 的 `offload` 开关解决的正是这个「谁在什么时候占多少显存」的问题，而 OOM 与步时延都由这张账决定。

本题给你一张**已经写好的调度表**——每模型的字节数、每阶段用哪个模型/跑哪台设备/是否训练、以及每个阶段每条流水线的常驻标记——要你算出账本，不做任何真实前向。

## 模型与口径

设 $M$ 个模型、$P$ 个阶段、$D$ 台设备。约定：

- `resident[p][m] == 1` 表示模型 $m$ 的**完整训练态**（$\text{param}_m+\text{opt}_m$）此刻就在 `model_device[m]` 上；$0$ 表示整块回收到 CPU 侧（offload）。优化器态与参数**同进同出**，这就是框架里整块 offload 的口径；推理模型的 `opt` 记 $0$。
- 激活只在**被更新**的那一步瞬时出现：阶段 $p$ 若 `phase_train[p]==1`，则 `phase_device[p]` 这台设备额外摊入 $\text{act}_{\text{phase\_model}[p]}$。推理阶段按 no-grad 分块处理，激活不计（KV cache 归 `kv-cache-memory`）。

于是阶段 $p$ 在设备 $d$ 上的占用为

$$\text{occ}[p][d]=\!\!\sum_{m:\ \text{model\_device}[m]=d,\ \text{resident}[p][m]=1}\!\!(\text{param}_m+\text{opt}_m)\ +\ \mathbb{1}[\text{phase\_train}[p]=1\wedge \text{phase\_device}[p]=d]\cdot \text{act}_{\text{phase\_model}[p]}$$

## 规则

输入九个量：`models` $(M,3)$ int64（列依次为 `param_bytes, opt_bytes, act_bytes`）、`model_device` $(M,)$、`phase_model` $(P,)$、`phase_device` $(P,)$、`phase_train` $(P,)\in\{0,1\}$、`resident` $(P,M)$ 0/1、`phase_time` $(P,)$、标量 `hbm_cap`、标量 `num_devices` $=D$。

1. **峰值**：`peak[d]` $=\max_p$ `occ[p][d]`（无阶段时为 $0$）。
2. **搬运量**：时间线以阶段 0 的配置为**稳态**，且一步结束必须回到同一稳态（否则下一步的生成阶段起点就错了）。于是模型 $m$ 的翻转次数
   $$\text{moves}_m=\#\{p\in[1,P): \text{resident}[p][m]\ne\text{resident}[p-1][m]\}+\mathbb{1}[\text{resident}[P-1][m]\ne\text{resident}[0][m]]$$
   `offload[model_device[m]]` $+=$ $\text{moves}_m\cdot(\text{param}_m+\text{opt}_m)$。注意最后这项「收尾翻转」：步末状态若与稳态不同，就要**再搬一次**回去。
3. **空闲窗口**：`idle_time[d]` $=\sum_p \text{phase\_time}[p]\cdot\mathbb{1}[\text{phase\_device}[p]\ne d]$；`max_idle[d]` $=$ 该设备**最长一段连续**空闲阶段的时长和（一步之内，不跨步回绕）。
4. **可行性**：`feasible[d]` $=$ (`peak[d] <= hbm_cap`)。

返回五元组：`peak` int64 $(D,)$、`offload` int64 $(D,)$、`idle_time` int64 $(D,)$、`max_idle` int64 $(D,)$、`feasible` bool $(D,)$。

## 示例

4 个模型（actor/critic/ref/reward）、6 个阶段、2 台设备：

```
models(行序=actor,critic,ref,reward) = [[400,2000,300],[400,2000,250],[400,0,120],[400,0,110]]
model_device = [0, 1, 0, 1]                      # actor+ref 在 GPU0，critic+reward 在 GPU1
阶段: p0 生成(actor,GPU0,infer,t=5)  p1 打分(reward,GPU1,infer,t=2)
      p2 价值(critic,GPU1,infer,t=2) p3 old_logp(actor,GPU0,infer,t=2)
      p4 ref_logp(ref,GPU0,infer,t=2) p5 actor更新(actor,GPU0,train,t=4)
phase_model=[0,3,1,0,2,0]  phase_device=[0,1,1,0,0,0]  phase_train=[0,0,0,0,0,1]
phase_time =[5,2,2,2,2,4]  hbm_cap=4000  num_devices=2

resident = [[1,0,0,1],      # 生成：actor 常驻，critic/ref 在 CPU，reward 在 GPU1
            [1,0,0,1],      # 打分
            [1,1,0,1],      # 价值：换入 critic
            [1,1,0,1],      # old_logp
            [1,1,1,0],      # ref_logp：换入 ref、offload reward
            [1,1,1,0]]      # actor 更新
```

占用矩阵（每行 = 设备 `[GPU0, GPU1]`）：

```
p0 [2400,  400]   p1 [2400,  400]   p2 [2400, 2800]
p3 [2400,  2800]  p4 [2800, 2400]   p5 [3100, 2400]   # p5 = 2400+400+act300
→ peak      = [3100, 2800]
→ offload   = [800, 5600]     # critic 翻 2 次×2400、ref 2×400、reward 2×400，actor 0 次
→ idle_time = [4, 13]         # 总时长 17；GPU0 忙 5+2+2+4=13，GPU1 忙 2+2=4
→ max_idle  = [4, 8]          # GPU1 的空闲被打断成 5 和 2+2+4=8 两段
→ feasible  = [True, True]    # hbm_cap 改成 3000 则变 [False, True]
```

## 约束

- CPU；$1\le M\le 16$、$0\le P\le 64$、$1\le D\le 8$；所有字节与时间为非负 int64；`resident` 只取 $0/1$；`model_device`、`phase_model`、`phase_device` 的下标必须在合法范围内。
- 判定：前四个用 `torch.equal`，`feasible` 用 `torch.equal`（bool 张量）。
- 退化不变量：`resident` 全 $1$ $\Rightarrow$ `offload` 全 $0$；$D=1$ $\Rightarrow$ `idle_time` 与 `max_idle` 全 $0$；`phase_train` 全 $0$ $\Rightarrow$ 任何设备的 `peak` 都不含 `act`；$P=0$ $\Rightarrow$ 四个整数输出全 $0$、`feasible` 全 `True`；`hbm_cap=0` 且存在常驻模型 $\Rightarrow$ `feasible` 全 `False`。
- 分工：`training-state-memory` 算**单模型**每参数 16 字节的训练态与 ZeRO 分片；`distributed-zero-offload` 算**单模型**卸载换 PCIe 带宽的瓶颈；本题算的是**多模型 × 多阶段**的共卡时间线（谁在何时占多少、搬了多少、谁在闲着）。优势/损失公式在 `to-*`，token 负载重排在 `rl-rollout-shard-repack`。
