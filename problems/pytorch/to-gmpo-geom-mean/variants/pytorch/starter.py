"""GMPO（callsys/GMPO, arXiv:2507.20673）token 几何平均策略梯度账 -- 你的实现。

官方 train_zero_math_gmpo.py 提供三方账：compute_monte_carlo_advantages 把
token 级奖励折成轨迹级优势（base / grpo std 归一 / drgrpo 不归一），
learning_step 的六路 pg-loss 分支（gmpo 家族四变体 + grpo_clip_wider +
标准 PPO-clip else 兜底），以及 KL token 奖励组装与 eos 注入。请实现下面
的 solve；只允许 import torch（判题环境可用，张量一律 float64 构造）。

参数
----
mode : str
    "advantage"（默认）| "pg" | "kl_reward"。
rewards : list[list[float]]
    mode="advantage" 用：token 级奖励 [batch*num_samples, seq]。
num_samples : int
    每条 prompt 的采样数（组大小）。
critic_type : None | "grpo" | "drgrpo"
    "grpo" 时除以组内 std（torch 默认 unbiased，n-1 分母）+ 1e-8；
    "drgrpo"/None 不归一。
variant : str
    mode="pg" 用："gmpo" | "gmpo_noclip" | "gmpo_seqclip" |
    "gmpo_without_norm" | "grpo_clip_wider" | "grpo"（else 兜底）。
new_logps, old_logps, response_masks : list[list[float]]
    mode="pg" 用：[1, seq]（官方仅单序列 mini-batch 可跑，batch>1 是
    官方原生 RuntimeError）；response_masks 取值 0/1（转 bool）。
advantage : list[float]
    轨迹级优势 [1]。
loss_masks : list[float]
    序列级 loss 权重 [1]（仅 grpo 系参与 (pg_loss * loss_masks).mean()）。
cliprange : float
    默认 0.2；gmpo_noclip 忽略此参数（内定 1000）。
aggregator : str
    "mean"（除 mask.sum()）| "sum"（drgrpo 风格，除 constant_normalizer）。
constant_normalizer : float
    aggregator="sum" 时必填正数。
logps, ref_logps : list[list[float]]
    mode="kl_reward" 用：策略/参考 logps [batch, seq]。
final_rewards : list[list[float]]
    [batch, 1] 结果奖励（内部 squeeze）。
coef : float
    KL 系数。

返回
----
mode="advantage"：优势 list（长度 batch*num_samples）。
mode="pg"：{"loss": ..., "stats": {...}}。gmpo 家族 loss 是向量 list
    （官方不做 masked mean）；grpo 系 loss 是标量 float。stats 三键
    logprobs_diff_max / logprobs_diff_min / zero_pg_loss_count 各一条。
mode="kl_reward"：{"rewards": ..., "eos_indices": ...}。

账本规则（官方逐条钉死）
------------------
advantage：rewards.sum(-1) 折成轨迹级 -> view(-1, num_samples).mean(1)
    为组均值 -> repeat_interleave 回每 token -> advantages = rewards -
    values；仅 critic_type="grpo" 再除 (std + 1e-8)。
gmpo：sgn = -1 if advantage >= 0 else 1（**>= 边界进 -1**）；diff 翻转
    -> clamp(-ε, +ε) -> max(翻转 diff, clamp)（保守截断）-> 翻回；
    ratio = exp(diff_max[mask].sum() / mask.sum())（token 几何平均）；
    loss = -adv * ratio（向量）。batch>1 时张量布尔化 RuntimeError。
gmpo_noclip：同账，cliprange 内定 1000（忽略入参）。
gmpo_seqclip：diff 先对 mask 求 sum 成**标量**再翻转-clamp-max；ratio =
    exp(diff_max / mask.sum())；stats 的 amax==amin（标量广播）。
gmpo_without_norm：ratio = exp(diff_max[mask].sum())，**不除 mask.sum()**。
grpo_clip_wider：token 级 PPO-clip 但 clamp(0.67, 1.49) 硬编码。
grpo（else）：ratio = exp(diff) 逐 token；pg_losses2 = -adv *
    clamp(ratio, 1-ε, 1+ε)；逐位 max -> masked_aggregator(axis=1) ->
    (pg_loss * loss_masks).mean() 成标量。
kl_reward：kl_rewards = -coef * (logps - ref_logps) * mask；eos_indices
    = 每行 mask 内最大下标（**全 False 行 = -1**）；rewards[行, eos] +=
    final_rewards.squeeze()——-1 是负索引，注入该行最后一个位置。

异常
----
mode/variant/critic_type/aggregator 未注册、参数类型不合法 -> ValueError；
batch>1 的 pg、rewards 总数不被 num_samples 整除 -> 官方 RuntimeError/
torch 异常原样传播。
"""
import torch


def solve(mode="advantage", rewards=None, num_samples=None, critic_type=None,
          variant=None, new_logps=None, old_logps=None, response_masks=None,
          advantage=None, loss_masks=None, cliprange=0.2, aggregator="mean",
          constant_normalizer=None, logps=None, ref_logps=None,
          final_rewards=None, coef=None):
    raise NotImplementedError(
        "TODO: 实现 advantage / pg / kl_reward 三入口")
