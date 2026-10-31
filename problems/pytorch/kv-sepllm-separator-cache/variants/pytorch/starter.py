import torch


def solve(token_ids, sep_ids, keys, values, init_cache_size=4, sep_cache_size=32,
          local_size=224, boundary=0):
    """SepLLM（ICML'25）训练无关 KV 压缩的单步，USE_MAX_SEP_CACHE=True 口径。
    token_ids:(L,) int64；sep_ids:分隔符 id 集合；keys/values:逐层 (H,L,D) 同 dtype（float32/float64）；
    init_cache_size/sep_cache_size/local_size 正整数；boundary 为分隔符区右边界，0 表示首次压缩。
    保留 [0,init) ∪ 分隔符位置 ∪ [L-local,L)，分隔符位置 = [init,boundary) 加上过去窗口
    [boundary, L-local) 里 id 属于 sep_ids 的位置；超过 sep_cache_size 时只留位置最大的 s 个。
    返回 dict：kept:(n,) int64 原始下标升序 / token_ids:(n,) int64 压缩后 id /
    keys,values:逐层 (H,n,D) 按 kept 聚集 / boundary:int = init+min(分隔符数,sep_cache_size)。
    内部 float64，返回输入 dtype。seq_len-local_size<=boundary（上游 assert）及其他非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
