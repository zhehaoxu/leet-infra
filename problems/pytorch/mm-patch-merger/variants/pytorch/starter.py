import torch


def solve(image_embeds, grid_h, grid_w):
    """Qwen2-VL PatchMerger 空间合并：(gh*gw, D)->(gh/2*gw/2, 4D)，
    每块按 左上→右上→左下→右下 拼接 4 个 patch。禁止 einops。"""
    # TODO
    raise NotImplementedError
