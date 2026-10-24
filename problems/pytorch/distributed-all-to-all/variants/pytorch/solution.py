import torch


def solve(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    W = len(inputs)
    # 网格 (src, dst, k) = stack 后 reshape；转置成 (dst, src, k) 再展平即 all-to-all
    grid = torch.stack([t.reshape(-1) for t in inputs], dim=0)   # (W, W*k)
    k = grid.shape[1] // W
    grid = grid.reshape(W, W, k).transpose(0, 1)                 # (dst, src, k)
    out = grid.reshape(W, W * k)
    return [out[j].clone() for j in range(W)]
