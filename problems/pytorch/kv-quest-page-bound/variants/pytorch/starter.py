import torch


def solve(q, k, page_size=16, page_budget=1):
    """Quest 页级 query-aware 上界打分与预算选页（2406.10774，CPU 参考口径）。
    q:(Lq,H,D)、k:(Lkv,H,D) 同 dtype（float32/float64），NHD 布局；只支持 decode 单 query，Lq 必须为 1。
    页数 P=ceil(Lkv/page_size)>=2，页内不足按输入 dtype 的 finfo.min 补齐。
    sign:(H,D)=where(q>0,+1,-1)（零归负）；page_max:(H,P,D)=(k*sign) 的页内 amax；
    scores:(H,P-1)=(q*sign)@page_max^T 去掉最后一列；selected:(H,K+1) int64 升序，
    K=min(page_budget,P)-1 个页号按分数降序选出（同分取小页号），末尾恒接 P-1。
    内部 float64，返回输入 dtype。返回 dict：sign / page_max / scores / selected。
    非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
