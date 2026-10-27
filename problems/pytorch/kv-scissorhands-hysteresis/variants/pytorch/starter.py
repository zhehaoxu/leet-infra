import torch


def solve(scores, tau, patience):
    """ScissorHands 时间一致性滞回驱逐。
    scores:(S,L) float32 逐步重要性分；tau 为阈值；patience=m 为连续低分步数门槛。
    维护逐 token 的「连续低分计数」：score<tau 加 1、否则清零；
    当计数首次达到 m 时该 token 在第 s 步被驱逐（此后不再回返）。
    返回 (evict_step:(L,) int64，首次被驱逐的步，未驱逐为 -1,
          keep:(S,L) bool，keep[s] 为第 s 步处理完驱逐后的存活掩码)。
    patience<1 抛 ValueError。阈值为严格小于（score==tau 不算低分）。"""
    # TODO
    raise NotImplementedError
