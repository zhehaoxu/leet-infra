import torch


def solve(x, media, media_locations, wq, wk, wv, wo, gate, n_heads, eps=1e-5):
    """中文契约 / English contract
    x: [B,T,D]; media: [B,M,K,D]；CPU float32/float64，同 dtype / same dtype.
    media_locations: bool [B,T]，媒体出现标记，非 padding；每行 True 数 <= M /
        media occurrence markers, not padding; at most M True entries per row (zero allowed).
    wq,wk,wv,wo: [D,D]，同 dtype/device，投影 x @ W.T / same dtype/device; x @ W.T.
    B=1..3, T=1..7, M=1..4, K=1..5, D=1..12.
    浮点值有限；x 绝对值 <=10，media 绝对值 <=1000，权重绝对值 <=4 /
        finite floats; abs(x)<=10, abs(media)<=1000, abs(weights)<=4.
    gate: Python 有限实标量 [-20,20] / finite Python real scalar in [-20,20].
    n_heads: 正 Python int，D 可整除，否则 ValueError（bool 非法） /
        positive Python int dividing D, else ValueError (bool excluded).
    eps: Python 实数 [1e-5,1]，默认 1e-5 / Python real in [1e-5,1], default 1e-5.
    其余输入合法 / All other inputs are valid.
    LN: 仅 x 的末维，无 affine，biased variance；media 不 LN /
        last dimension of x only, no affine, biased variance; no LN on media.
    返回 / Returns (out [B,T,D], attention [B,H,T,M*K], visible bool [B,T,M*K]);
        H=n_heads，浮点输出同 dtype/device / floating outputs preserve dtype/device.
    无 cache，只看最近媒体，标记本身即时可见；首图前 attention 严格零 /
        no cache; immediate media only, including its marker; exact zero attention before first.
    支持非连续张量，不修改输入 / accept noncontiguous tensors; mutate no inputs.
    禁止 SDPA/multi_head_attention；可用 layer_norm/softmax /
        no SDPA/multi_head_attention; layer_norm/softmax allowed.
    """
    raise NotImplementedError
