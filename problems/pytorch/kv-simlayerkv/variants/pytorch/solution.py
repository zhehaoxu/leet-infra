"""参考答案：SimLayerKV 懒惰层二值保留（张量切片版，判题用）。"""

import torch


def solve(last_attn, n_initial, n_recent, n_last, lazy_threshold):
    N, L = last_attn.shape
    mask = torch.ones(N, L, dtype=torch.bool)
    ratios = torch.zeros(N, dtype=torch.float32)
    # 源码 quirk：threshold==1.0 或序列过短 → 整层不压缩
    if lazy_threshold == 1.0 or L <= n_initial + n_recent + n_last:
        return mask, ratios
    scores = (last_attn[:, :n_initial].sum(dim=1)
              + last_attn[:, L - n_recent:].sum(dim=1))
    lazy = scores > lazy_threshold
    if bool(lazy.any()):
        m = mask[lazy]
        m[:, :] = False
        m[:, :n_initial] = True
        m[:, L - (n_recent - n_last):] = True
        mask[lazy] = m
        ratios[lazy] = float(L - n_initial - (n_recent - n_last)) / float(L)
    return mask, ratios
