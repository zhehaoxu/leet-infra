import torch


def solve(draft_probs, target_probs, draft_tokens, u_accept, u_resample):
    """Speculative decoding 拒绝采样：逐 draft token 以 min(1,p/q) 接受，
    拒绝则从 normalize(max(0,p-q)) 重采样并停止；全接受则从 target_probs[γ] 取 bonus。
    采样=CDF 反查（最小 k 使 cumsum>=u）。返回 torch.long 一维 token 序列。"""
    # TODO
    raise NotImplementedError
