import torch


def solve(q, k, v, cos, sin, keep, group_size=1):
    """CapKV：平均未来 RoPE 的 query anchor + 加权 leverage score（arXiv 2604.25975）。
    q:(Lq,Hq,D)，k/v:(T,Hkv,D) 同 dtype（float32/float64），Hq=Hkv*group_size；cos/sin:(P,D) RoPE 表，D 为正偶数。
    前 n_sink=4 个 token 不参与打分，但永远保留。tau=5.0，epsilon=1e-6 加在 sqrt(w+epsilon) 里。
    statistic/weights/leverage:(Hq,T-4)；scores:(Hkv,T)，前 4 列填跨 head 全局 max+1；
    kept:(Hkv,keep) int64 每行升序，按分数降序选出（同分取小下标）。
    内部 float64，返回输入 dtype；不允许改输入。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
