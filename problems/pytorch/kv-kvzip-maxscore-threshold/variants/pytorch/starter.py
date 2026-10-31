import torch


def solve(attn_weights, ratio=0.3, level="pair"):
    """KVzip 跨查询聚合 + 全局阈值裁剪（2505.23416）。
    attn_weights:(L,H_kv,G,Q,N) float32/float64，已 softmax、已去掉 sink 前缀。
    scores:(L,H_kv,N)=跨 group 与 query 的 amax；pair 模式把全部层 flatten 成 T 个分数降序排序，
    n=max(int(T*ratio)-1,0) 是**下标**，threshold=第 n 大，valid=(scores>threshold)（严格大于）；
    pair-uniform 模式改成每 (layer,head) 取 int(N*ratio) 个最高分（同分取小列号），threshold=0。
    返回 dict：scores/valid/keep_count/threshold/threshold_index/keep_ratio。内部 float64。
    非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
