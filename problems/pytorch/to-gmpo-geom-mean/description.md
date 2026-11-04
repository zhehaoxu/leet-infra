# GMPO：token 几何平均优势与符号翻转保守截断（MC 优势三分支 + 六路 pg-loss + KL token 奖励注入）

## 题意

GMPO（arXiv:2507.20673, 官方仓 `callsys/GMPO`）把 GRPO 的「组内算术平均优势」换成对 token 级对数比做**几何平均**（log 空间求和再取指数），并按 advantage 符号翻转做保守截断。本题取 `train_zero_math_gmpo.py` 的三方账，全部显式入参：

```text
mode="advantage": rewards(token 级 2-D) + num_samples + critic_type(None|"grpo"|"drgrpo") -> 优势 list
mode="pg":        variant + new_logps + old_logps + response_masks + advantage + loss_masks
                  (+cliprange + aggregator + constant_normalizer) -> {"loss", "stats"}
mode="kl_reward": logps + ref_logps + response_masks + final_rewards + coef -> {"rewards", "eos_indices"}
```

## 账本规则（官方逐条钉死）

`mode="advantage"`（官方 `compute_monte_carlo_advantages` L407-421）：

```text
rewards = rewards.sum(-1)                      # token 级奖励折成轨迹级
values = rewards.view(-1, num_samples).mean(1) # 组内算术均值
values = values.repeat_interleave(num_samples) # 回到每 token
advantages = rewards - values
critic_type="grpo": 再除以组内 std（torch 默认 unbiased、n-1 分母）+ 1e-8
critic_type="drgrpo"/None: 不归一，advantages 原样返回
```

`mode="pg"`（官方 `learning_step` 六分支 L673-831）。gmpo 家族四变体共用
「符号翻转 + 保守截断」骨架，差别只在 diff 的折叠方式与归一：

```text
gmpo:
  sgn = -1 if advantage >= 0 else 1        # >= 边界进 -1
  diff = new - old; 翻转 -> clamp(-ε, +ε) -> max(翻转 diff, clamp) -> 翻回
  ratio = exp( diff_max[mask].sum() / mask.sum() )   # token 几何平均
  pg_losses = -advantage * ratio            # 向量，loss 不做 masked mean
gmpo_noclip:  同 gmpo，cliprange 内定 1000（忽略入参）
gmpo_seqclip: diff 先对 mask 求 sum 成标量，再翻转-clamp-max；
              ratio = exp(diff_max / mask.sum())；stats 的 amax==amin==sum
gmpo_without_norm: ratio = exp( diff_max[mask].sum() )，不除 mask.sum()
```

grpo 系两分支是「标准 PPO-clip 骨架 + 差异化的 clamp 域」：

```text
grpo_clip_wider: clamp(0.67, 1.49) 硬编码（入参 cliprange 不参与 clamp）
grpo（else 兜底）:
  ratio = exp(diff) 逐 token
  pg_losses2 = -adv * clamp(ratio, 1-ε, 1+ε)
  pg_loss_max = max(pg_losses, pg_losses2) 逐位
  pg_loss = masked_aggregator(pg_loss_max, mask, axis=1)
            （aggregator="mean" 除 mask.sum()；"sum" 除 constant_normalizer）
  loss = (pg_loss * loss_masks).mean()      # 标量
```

stats 三键每 mini-batch 各 append 一条：`logprobs_diff_max` / `logprobs_diff_min`
是 `diff * mask` 的全局 amax/amin（seqclip 的 diff 是标量 → amax==amin）；
`zero_pg_loss_count` 是 `(pg_loss_max == 0).sum()`（gmpo 系目标是 pg_losses 向量）。

官方怪癖（测试钉死）：advantage 是张量，`sgn = -1 if adv >= 0 else 1` 在
batch>1 时布尔化 RuntimeError；grpo 系的 `-adv[batch] * ratio[batch, seq]`
广播失败——**两个 pg 家族都只支持单序列 mini-batch**。

`mode="kl_reward"`（官方 L594-605）：

```text
kl_rewards = -coef * (logps - ref_logps) * mask
eos_indices = 每行 mask 内最大下标（全 False 行得 -1）
rewards[行, eos_indices] += final_rewards.squeeze()
```

全 False 行的 eos=-1 是**负索引**：final reward 注入该行**最后一个位置**
（pytorch 负下标语义），而非报错。

## 关键怪癖（官方口径）

- advantage 的 grpo 归一用 `std(dim=1)` **默认 unbiased**（n-1 分母），与 Dr.GRPO 的 n 分母不同；`+1e-8` 在分母上；
- gmpo 的 sgn 翻转在 clamp **之前**、恢复在 max **之后**：`max(sgn·diff, clamp(sgn·diff))` 是「保守截断」——正方向全保、负方向只保留被 clamp 放大的部分；
- ratio 的几何平均在 **log 空间求和**后取 exp，除法对象是 `mask.sum()`（有效 token 数），不是序列长度；
- gmpo 家族 loss 是**向量**（每序列一个 ratio），不做 masked mean——与 grpo 系标量化路径不同；
- noclip 的 ε=1000 是**分支内定值**，不读 args.cliprange；
- seqclip 的 stats amax==amin：diff 折叠成标量后 `amax(标量*mask)` 与 `amin` 相等；
- batch>1 的 RuntimeError 是官方原生行为（张量布尔化 / 广播失败），不是本题校验。

## 输入约定

张量一律 float64 构造；`response_masks` 取值 0/1（内部转 bool）；`final_rewards`
是 `[batch, 1]`（内部 squeeze）；`aggregator="sum"` 必须给正数
`constant_normalizer`（官方 drgrpo 用 generate_max_length）。
