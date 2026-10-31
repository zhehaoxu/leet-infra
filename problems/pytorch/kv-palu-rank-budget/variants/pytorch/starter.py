import torch


def solve(fisher, param_ratio_target, head_group_size, lr_group_dims, block_size=32):
    """Palu 分组秩预算分配（2407.21118，Fisher 加权 + 块对齐）。
    fisher:(G*head_group_size, lr_group_dims) 浮点；G=fisher.shape[0]//head_group_size，
    组 Fisher=该组整个块的均值；total_rank=G*lr_group_dims，target_rank=total_rank*ratio。
    rank_float:(G,)=target_rank*f/Σf；rank_floor:(G,) int64=min(lr_group_dims,floor(rank_float))。
    rank_topup：按余数升序轮转 +1，已满组跳过，直到 target_rank-Σrank_floor<=0（故加 floor 次）。
    rank_aligned:(G,) int64=max(1, round_half_even(rank_topup/block_size))*block_size，不再截断。
    rank_sum:0 维 int64=Σrank_aligned；stored_ratio=rank_sum/total_rank。
    内部 float64，浮点输出用输入 dtype；不修改输入；非法输入抛 ValueError。返回 dict 6 键。"""
    # TODO
    raise NotImplementedError
