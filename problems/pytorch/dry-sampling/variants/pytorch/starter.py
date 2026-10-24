import torch


def solve(logits, history, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n):
    # TODO: DRY 惩罚（反向 Z-algorithm；返回惩罚后的 logits，不做采样；无 sequence breakers）。
    # last = history if dry_penalty_last_n<0 else history[-dry_penalty_last_n:]；N=len(last)。
    # 早退：mult==0 / base<1 / last_n==0 / N<=allowed → 原样返回。
    # 对每个 shift k：L_k = rev 与 rev[k:] 的最长公共前缀；若 L_k>=allowed 则罚 token last[N-k]（取最大 L_k）。
    # logits[t] -= mult * base**max(R-allowed,0)。
    raise NotImplementedError
