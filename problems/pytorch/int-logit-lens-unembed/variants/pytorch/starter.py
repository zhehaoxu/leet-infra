import torch


def solve(hiddens, lnf_w, lnf_b, unembed, apply_norm=True, eps=1e-5):
    """转写 logit lens 的简化读出账（nostalgebraist 口径：中间层隐态直接过
    final LayerNorm + unembed 矩阵读词表；真实模型 forward 不出，隐态全显式入参）。

    参数
    ----
    hiddens : [L, D]   逐层（最后位置）隐态
    lnf_w   : [D]      final LayerNorm 的 weight
    lnf_b   : [D]      final LayerNorm 的 bias
    unembed : [D, V]   unembedding 矩阵
    apply_norm : bool  False 时跳过 LayerNorm（退化为 raw hidden 读出）
    eps     : float    LayerNorm 的方差加项（缺省 1e-5，须 > 0）

    每层 l：
        n = (h - mean(h)) / sqrt(var(h) + eps) * lnf_w + lnf_b   # var 为有偏（除 D）
        logits[l] = n @ unembed
        argmax[l] = 最小下标的最大分量（并列取小）
    final_argmax = argmax[L-1]
    agree[l]     = (argmax[l] == final_argmax)
    rank_of_final[l] = 1 + #{v : logits[l][v] > logits[l][final_argmax]}
                       （final 层恒为 1；严格大于计数，并列不影响）
    top_logit[l] = max(logits[l])

    返回
    ----
    dict：logits [L,V]（嵌套 list）、argmax [L]、final_argmax、agree [L]、
          rank_of_final [L]、top_logit [L]
    """
    raise NotImplementedError
