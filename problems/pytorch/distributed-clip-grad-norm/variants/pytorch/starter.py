import torch


def solve(shards: list[torch.Tensor], max_norm: float, norm_type: float = 2.0):
    # TODO: 全局 p-范数 = (Σ_r Σ_i |g|^p)^(1/p)；scale=min(1, max_norm/norm)；每分片同乘 scale
    #   返回 (clipped_list, total_norm)；不要各 rank 独立裁剪，也不要修改输入
    raise NotImplementedError
