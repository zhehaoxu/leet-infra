import torch


def solve(visual, latents, visual_mask, wq, wk, wv, wo, n_heads, eps=1e-5):
    """中文契约 / English contract
    visual: [B,N,D]; latents: [B,K,D]; CPU float32/float64，同 dtype / same dtype.
    visual_mask: bool [B,N]，True 可见 / True means visible.
    wq,wk,wv,wo: [D,D]，同 dtype/device；投影 x @ W.T / same dtype/device; x @ W.T.
    B=1..3, N=0..12, K=1..6, D=1..12.
    浮点值有限；visual/latents 绝对值 <=10，权重绝对值 <=4 /
        finite floats; abs(visual/latents)<=10, abs(weights)<=4.
    n_heads: 正 Python int，D 可整除，否则 ValueError（bool 非法） /
        positive Python int dividing D, else ValueError (bool excluded).
    eps: Python 实数 [1e-5,1]，默认 1e-5 / Python real in [1e-5,1], default 1e-5.
    其余输入合法 / All other inputs are valid.
    LN: 仅末维 D，无 affine，biased variance / last D only, no affine, biased variance.
    返回 / Returns (output [B,K,D], attention [B,H,K,N+K]), 同浮点 dtype/device /
        same floating dtype/device. H=n_heads；latent K/V 永远可见 / always visible.
    支持非连续张量，不修改任何输入 / accept noncontiguous tensors; mutate no inputs.
    禁止 SDPA/multi_head_attention；可用 layer_norm/softmax /
        no SDPA/multi_head_attention; layer_norm/softmax allowed.
    """
    raise NotImplementedError
