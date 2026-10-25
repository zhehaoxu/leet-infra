import torch


def solve(image, patch):
    """ViT patchify：(B,C,H,W) -> (B, (H/p)*(W/p), C*p*p)。
    token 按块行主序、token 内按 (通道,块内行,块内列) 展平。禁止 unfold/im2col。"""
    # TODO
    raise NotImplementedError
