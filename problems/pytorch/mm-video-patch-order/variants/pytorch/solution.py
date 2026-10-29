import torch


def solve(video, temporal_patch_size, patch_size, merge_size):
    tau, p, m = temporal_patch_size, patch_size, merge_size
    T, C, H, W = video.shape
    if H % (p * m) or W % (p * m):
        raise ValueError("H and W must be divisible by patch_size * merge_size")

    padding = (-T) % tau
    if padding:
        video = torch.cat((video, video[-1:].expand(padding, -1, -1, -1)), dim=0)
    gt, gh, gw = video.shape[0] // tau, H // p, W // p
    patches = (
        video.reshape(gt, tau, C, gh // m, m, p, gw // m, m, p)
        .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .reshape(gt * gh * gw, C * tau * p * p)
    )

    t, block_h, block_w, inner_h, inner_w = torch.meshgrid(
        torch.arange(gt, device=video.device),
        torch.arange(gh // m, device=video.device),
        torch.arange(gw // m, device=video.device),
        torch.arange(m, device=video.device),
        torch.arange(m, device=video.device),
        indexing="ij",
    )
    coords = torch.stack((t, block_h * m + inner_h, block_w * m + inner_w), dim=-1)
    return patches, (gt, gh, gw), coords.reshape(-1, 3)
