"""GMPO（arXiv:2507.20673, callsys/GMPO）token 几何平均策略梯度账。

GMPO 把 GRPO 的「组内算术平均优势」换成对 token 级对数比做**几何平均**
（log 空间求和再取指数），并按 advantage 符号翻转做保守截断。官方
train_zero_math_gmpo.py 提供三方账：compute_monte_carlo_advantages 的
token 奖励 -> 轨迹均值优势（base / grpo std 归一 / drgrpo 不归一）、
learning_step 的六路 pg-loss 分支（gmpo 家族四变体 + grpo_clip_wider +
标准 PPO-clip 的 else 兜底）、以及 KL token 奖励的组装与 eos 注入。
本题全部显式入参，考纯账本。

solve(mode=...) 三个入口（官方逐条钉死）：

    mode="advantage"（compute_monte_carlo_advantages L407-421）：
        rewards (token 级 2-D) -> advantages (list)。
        先 rewards.sum(-1) 折成轨迹级奖励；values = view(-1, num_samples)
        的组内算术均值，repeat_interleave 回每个 token；advantages =
        rewards - values。critic_type="grpo" 再除以组内 std（torch 默认
        **unbiased**，n-1 分母）+ 1e-8；critic_type="drgrpo"（及 None）
        **不归一**——advantages 原样返回。

    mode="pg"（learning_step 六分支 L673-831）：variant + new_logps +
        old_logps + response_masks + advantage + loss_masks (+cliprange +
        aggregator + constant_normalizer) -> {"loss", "stats"}。
        gmpo：sgn_advantage = -1 if advantage >= 0 else 1（**>= 边界进
        -1**；advantage 是张量，batch>1 时布尔化 RuntimeError——官方
        实际只能单序列 mini-batch）；diff = new - old 翻转 -> clamp 到
        [-ε, +ε] -> max(翻转 diff, clamp)（**保守截断**）-> 翻回；
        ratio = exp( logprobs_diff_max[mask].sum() / mask.sum() )——
        **token 几何平均**；pg_losses = -advantage * ratio（**向量**，
        loss 不做 masked mean）；gmpo_noclip 同账但 cliprange **内定
        1000**（忽略入参）；gmpo_seqclip 的 diff 是 **先对 mask 求 sum
        的标量**再翻转-clamp-max（stats 的 amax/amin 因此相等）；gmpo_
        without_norm 的 ratio = exp(logprobs_diff_max[mask].sum())——
        **不除 mask.sum()**；grpo_clip_wider：标准 PPO-clip 骨架但
        clamp(0.67, 1.49) 硬编码；else（"grpo"）：token 级 ratio =
        exp(diff)、pg_losses2 = -adv * clamp(ratio, 1-ε, 1+ε)、逐位
        max、masked_aggregator(axis=1)（"mean" 除 mask.sum()，"sum"
        除 constant_normalizer）、再 (pg_loss * loss_masks).mean() 成
        标量。stats 三键 logprobs_diff_max / logprobs_diff_min /
        zero_pg_loss_count 各 append 一条（amax/amin 是 diff*mask 的
        全局极值 .item()；zero 计数是 (pg_loss_max == 0) 的 sum）。

    mode="kl_reward"（L594-605 token 奖励组装）：logps + ref_logps +
        response_masks + final_rewards + coef -> {"rewards", "eos_indices"}。
        kl_rewards = -coef * (logps - ref_logps) * mask；按行取 mask 内
        最大下标为 eos_indices（**全 False 行得 -1**）；rewards[行,
        eos] += final_rewards.squeeze()——全 False 行的 -1 是**负索引**，
        注入进该行**最后一个位置**。
"""
import torch

MODES = ("advantage", "pg", "kl_reward")
PG_VARIANTS = ("gmpo", "gmpo_noclip", "gmpo_seqclip", "gmpo_without_norm",
               "grpo_clip_wider", "grpo")
CRITIC_TYPES = (None, "grpo", "drgrpo")
AGGREGATORS = ("mean", "sum")


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_num(value, name):
    if not _is_num(value):
        raise ValueError(f"{name} must be a number")


def _check_2d(value, name):
    if not isinstance(value, list) or not value or \
            any(not isinstance(row, list) or not row for row in value):
        raise ValueError(f"{name} must be a non-empty 2-D list")
    if any(any(not _is_num(v) for v in row) for row in value):
        raise ValueError(f"{name} must contain only numbers")


def _check_1d(value, name):
    if not isinstance(value, list) or not value or \
            any(not _is_num(v) for v in value):
        raise ValueError(f"{name} must be a non-empty 1-D list of numbers")


def _masked_mean(x, mask, axis):
    return (x * mask).sum(axis) / mask.sum(axis)


def _masked_sum(x, mask, axis, constant_normalizer):
    return (x * mask).sum(axis) / constant_normalizer


def _compute_monte_carlo_advantages(rewards, num_samples, critic_type):
    """官方 compute_monte_carlo_advantages（L407-421，self.args 参数化）。"""
    rewards = rewards.sum(-1)
    # Compute monte carlo trajectory-level advantage
    values = rewards.view(-1, num_samples).mean(dim=1)
    values = values.repeat_interleave(num_samples, dim=0)
    advantages = rewards - values
    if critic_type == "grpo":
        # Additionally normalize by std.
        std_grouped_rewards = rewards.view(-1, num_samples).std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            num_samples, dim=0
        )
        advantages = advantages / (std_grouped_rewards + 1e-8)
    return advantages


def _stats_record(stats, logprobs_diff, mask, pg_loss, zero_target):
    """官方 stats 三键 append（每 mini-batch 一条）。"""
    stats["logprobs_diff_max"].append(
        torch.amax(logprobs_diff.detach() * mask).item()
    )
    stats["logprobs_diff_min"].append(
        torch.amin(logprobs_diff.detach() * mask).item()
    )
    stats["zero_pg_loss_count"].append(
        (zero_target == 0).detach().sum().item()
    )
    return pg_loss


def _pg_loss(variant, new_logps, mb_logps, mb_response_masks, mb_advantage,
             mb_loss_masks, cliprange, masked_aggregator):
    """官方 learning_step 六路 pg-loss 分支（L673-831）。"""
    stats = {"logprobs_diff_max": [], "logprobs_diff_min": [],
             "zero_pg_loss_count": []}
    if variant == "gmpo":
        low_cliprange = torch.tensor(-cliprange).to(new_logps)
        high_cliprange = torch.tensor(cliprange).to(new_logps)
        sgn_advantage = -1 if mb_advantage >= 0 else 1
        logprobs_diff = new_logps - mb_logps
        sgn_logprobs_diff = sgn_advantage * logprobs_diff
        sgn_logprobs_diff_clamp = torch.clamp(
            sgn_logprobs_diff, low_cliprange, high_cliprange)
        sgn_logprobs_diff_max = torch.max(
            sgn_logprobs_diff, sgn_logprobs_diff_clamp)
        logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max
        ratio = torch.exp(
            logprobs_diff_max[mb_response_masks].sum()
            / mb_response_masks.sum())
        pg_losses = -mb_advantage * ratio
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_losses, pg_losses)
        pg_loss = pg_losses
        loss = pg_loss
    elif variant == "gmpo_noclip":
        cliprange = 1000
        low_cliprange = torch.tensor(-cliprange).to(new_logps)
        high_cliprange = torch.tensor(cliprange).to(new_logps)
        sgn_advantage = -1 if mb_advantage >= 0 else 1
        logprobs_diff = new_logps - mb_logps
        sgn_logprobs_diff = sgn_advantage * logprobs_diff
        sgn_logprobs_diff_clamp = torch.clamp(
            sgn_logprobs_diff, low_cliprange, high_cliprange)
        sgn_logprobs_diff_max = torch.max(
            sgn_logprobs_diff, sgn_logprobs_diff_clamp)
        logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max
        ratio = torch.exp(
            logprobs_diff_max[mb_response_masks].sum()
            / mb_response_masks.sum())
        pg_losses = -mb_advantage * ratio
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_losses, pg_losses)
        pg_loss = pg_losses
        loss = pg_loss
    elif variant == "gmpo_seqclip":
        low_cliprange = torch.tensor(-cliprange).to(new_logps)
        high_cliprange = torch.tensor(cliprange).to(new_logps)
        sgn_advantage = -1 if mb_advantage >= 0 else 1
        logprobs_diff = (new_logps - mb_logps)[mb_response_masks].sum()
        sgn_logprobs_diff = sgn_advantage * logprobs_diff
        sgn_logprobs_diff_clamp = torch.clamp(
            sgn_logprobs_diff, low_cliprange, high_cliprange)
        sgn_logprobs_diff_max = torch.max(
            sgn_logprobs_diff, sgn_logprobs_diff_clamp)
        logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max
        ratio = torch.exp(logprobs_diff_max / mb_response_masks.sum())
        pg_losses = -mb_advantage * ratio
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_losses, pg_losses)
        pg_loss = pg_losses
        loss = pg_loss
    elif variant == "gmpo_without_norm":
        low_cliprange = torch.tensor(-cliprange).to(new_logps)
        high_cliprange = torch.tensor(cliprange).to(new_logps)
        sgn_advantage = -1 if mb_advantage >= 0 else 1
        logprobs_diff = new_logps - mb_logps
        sgn_logprobs_diff = sgn_advantage * logprobs_diff
        sgn_logprobs_diff_clamp = torch.clamp(
            sgn_logprobs_diff, low_cliprange, high_cliprange)
        sgn_logprobs_diff_max = torch.max(
            sgn_logprobs_diff, sgn_logprobs_diff_clamp)
        logprobs_diff_max = sgn_advantage * sgn_logprobs_diff_max
        ratio = torch.exp(logprobs_diff_max[mb_response_masks].sum())
        pg_losses = -mb_advantage * ratio
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_losses, pg_losses)
        pg_loss = pg_losses
        loss = pg_loss
    elif variant == "grpo_clip_wider":
        logprobs_diff = new_logps - mb_logps
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 0.67, 1.49)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_loss_max, pg_loss_max)
        pg_loss = masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
        pg_loss = (pg_loss * mb_loss_masks).mean()
        loss = pg_loss
    else:
        logprobs_diff = new_logps - mb_logps
        ratio = torch.exp(logprobs_diff)
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(
            ratio, 1.0 - cliprange, 1.0 + cliprange
        )
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        _stats_record(stats, logprobs_diff, mb_response_masks,
                      pg_loss_max, pg_loss_max)
        pg_loss = masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
        pg_loss = (pg_loss * mb_loss_masks).mean()
        loss = pg_loss
    return loss, stats


def _kl_token_reward(logps, ref_logps, response_masks, final_rewards, coef):
    """官方 KL token 奖励组装（L594-605）。"""
    kl_rewards = -coef * (logps - ref_logps) * response_masks
    rewards = kl_rewards.clone()
    indices = torch.arange(response_masks.size(1)).expand_as(response_masks)
    masked_indices = torch.where(
        response_masks, indices, torch.full_like(indices, -1))
    eos_indices = masked_indices.max(dim=1).values
    rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()
    return rewards, eos_indices


def solve(mode="advantage", rewards=None, num_samples=None, critic_type=None,
          variant=None, new_logps=None, old_logps=None, response_masks=None,
          advantage=None, loss_masks=None, cliprange=0.2, aggregator="mean",
          constant_normalizer=None, logps=None, ref_logps=None,
          final_rewards=None, coef=None):
    """统一入口（判题协议）：按 mode 分发到 advantage / pg / kl_reward。"""
    if mode == "advantage":
        _check_2d(rewards, "rewards")
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) \
                or num_samples < 1:
            raise ValueError("num_samples must be a positive int")
        if critic_type not in CRITIC_TYPES:
            raise ValueError(f"critic_type must be one of {CRITIC_TYPES}")
        r = torch.tensor(rewards, dtype=torch.float64)
        adv = _compute_monte_carlo_advantages(r, num_samples, critic_type)
        return adv.tolist()
    if mode == "pg":
        if variant not in PG_VARIANTS:
            raise ValueError(f"variant must be one of {PG_VARIANTS}")
        _check_2d(new_logps, "new_logps")
        _check_2d(old_logps, "old_logps")
        _check_2d(response_masks, "response_masks")
        _check_1d(advantage, "advantage")
        _check_1d(loss_masks, "loss_masks")
        _check_num(cliprange, "cliprange")
        if aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if aggregator == "sum":
            if not _is_num(constant_normalizer) or constant_normalizer <= 0:
                raise ValueError(
                    "constant_normalizer must be a positive number")
            cn = float(constant_normalizer)

            def masked_aggregator(x, mask, axis):
                return _masked_sum(x, mask, axis, cn)
        else:
            masked_aggregator = _masked_mean
        new_t = torch.tensor(new_logps, dtype=torch.float64)
        old_t = torch.tensor(old_logps, dtype=torch.float64)
        mask_t = torch.tensor(response_masks, dtype=torch.float64).bool()
        adv_t = torch.tensor(advantage, dtype=torch.float64)
        lm_t = torch.tensor(loss_masks, dtype=torch.float64)
        loss, stats = _pg_loss(variant, new_t, old_t, mask_t, adv_t, lm_t,
                               cliprange, masked_aggregator)
        return {"loss": loss.tolist(), "stats": stats}
    if mode == "kl_reward":
        _check_2d(logps, "logps")
        _check_2d(ref_logps, "ref_logps")
        _check_2d(response_masks, "response_masks")
        _check_2d(final_rewards, "final_rewards")
        _check_num(coef, "coef")
        logps_t = torch.tensor(logps, dtype=torch.float64)
        ref_t = torch.tensor(ref_logps, dtype=torch.float64)
        mask_t = torch.tensor(response_masks, dtype=torch.float64).bool()
        fin_t = torch.tensor(final_rewards, dtype=torch.float64)
        r, eos = _kl_token_reward(logps_t, ref_t, mask_t, fin_t, coef)
        return {"rewards": r.tolist(), "eos_indices": eos.tolist()}
    raise ValueError(f"Unsupported mode {mode!r}")
