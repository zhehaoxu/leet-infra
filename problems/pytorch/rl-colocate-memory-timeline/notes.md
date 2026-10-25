# 详解：多模型 colocate 的显存与阶段时间线账

## 出处与背景
- verl 把 actor/rollout/ref 装进同一个 `actor_rollout_ref` worker、critic 另起一个 worker，靠 `*_fsdp_config.offload`（`offload_model` / `offload_optimizer`）在阶段之间把参数与优化器态整块搬到 CPU；OpenRLHF 的 `--colocate` 与 DeepSpeed-Chat 的 ZeRO-3 + offload 是同一件事的两种写法。
- 一步 PPO 的阶段序列大致是：generation → reward → value → old_log_prob → ref_log_prob → advantage → actor/critic update。**只有 update 阶段需要优化器态与前向激活**，ref/reward 只在被问一次的那两个阶段才需要 onload——这正是"用搬运换显存"的全部来源。
- 本题把这套调度写成一张表，问的是账本：峰值、搬运字节、空闲窗口、可行性。所有量都是给定整数的加减乘，不涉及任何真实前向（真前向的显存公式在 `training-state-memory` / `distributed-pipeline-1f1b-memory`）。

## 四条必须钉住的口径
1. **整块进出**：`resident[p][m]=1` 表示 $\text{param}_m+\text{opt}_m$ 全在这台卡上，0 表示全不在。没有"参数在、优化器不在"的半吊子状态——所以搬运字节永远是 `moves × (param+opt)`，`offload` 与 `peak` 用的是同一个 `full` 向量。
2. **激活只跟 train 走、只落在 phase_device**：这是本题最容易写错的一行。向量化时若写成 `occ += train * act[phase_model]`（形状 $(P,1)$）会被广播成"每台卡都多背一份激活"，`peak` 直接算高；必须**按阶段把 act 加到 `phase_device[p]` 那一列**。参考解里这一处刻意用循环而不是广播，就是为了让这条约定不被动破。
3. **稳态闭合**：时间线以阶段 0 的配置为稳态，一步结束时必须回到它（下一步的生成阶段要能从同一状态起跑）。因此每个模型的翻转次数是"阶段间的差异数 $+$ 收尾回库的那一次"，结果是**偶数**——这条奇偶性是免费的自检（`check_offload_moves_are_even`）。忘记收尾 = 少算一次把 ref/critic 换回来的搬运。
4. **两种空闲口径**：`idle_time` 是总空闲（衡量"这台卡还能塞多少别的事"），`max_idle` 是**最长连续**空闲（衡量"能不能塞下一个完整的额外模型/一次预取"）。示例里 GPU1 总空闲 13 但最长连续只有 8——只看总数会高估可用窗口。按题面不跨步回绕。

## 可行性判据
`feasible[d] = peak[d] <= hbm_cap` 是逐设备的**严格不等式含等号**（等于上限算可行，和真实分配器一样按"刚好装满"处理）。返回向量而不是单一布尔，是因为共卡方案常出现"GPU0 溢出、GPU1 还有余量"的情形——只报 `all(feasible)` 会丢掉换卡方向的信息。真要"换个 assignment 是否可行"是搜索问题，不在本题范围。

## 常见坑
- 把 `peak` 算成 $\max_p\sum_d$（全局峰值）而不是逐设备峰值：多模型共卡的痛点恰恰是**某一台**爆掉。
- 用 `phase_model` 的设备而不是 `phase_device` 来摊激活（见口径 2）。
- 忘记推理模型 `opt=0`：给 ref/reward 记优化器态会让 `peak` 与 `offload` 双双虚高。
- $P=0$ 时 `occ.max(dim=0)` 在空张量上报错——必须显式回全零；此时 `feasible` 全 True。
- `hbm_cap=0` 且存在常驻模型时全 False，这是"表算"与"实际分配器"的差别：本题不做分配器余量。

## 与其他题族的分工
- `training-state-memory`：单模型每参数 16 字节的训练态与 ZeRO stage 1/2/3 分片后的每卡字节。
- `distributed-zero-offload`：单模型把优化器态/梯度卸到 CPU，算 GPU 显存、CPU 内存、PCIe 字节与是否成为带宽瓶颈。
- `distributed-pipeline-1f1b-memory`：流水并行下 1F1B 的激活驻留账。
- 以上都只涉及**一个模型**；本题的机制是**多个模型在同一批卡上按阶段表换入换出**，新增的量是"搬运次数/收尾回库""跨设备空闲窗口""逐设备可行性"。
- `rl-rollout-shard-repack` 管 token 负载在 rank 间怎么摊；本题管显存在设备-阶段平面上怎么摊。`perf-*` 的吞吐账是时间/算力，这里是容量与搬运。

## 复杂度
$O(P\cdot(M+D))$ 逐阶段累加，最长空闲窗口 $O(P\cdot D)$。oracle 用四重显式循环 + 状态序列展开，结构与参考解不同以便交叉验证。
