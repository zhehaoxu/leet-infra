import torch


def solve(x, window):
    """Swin window partition：(B,H,W,C)->(B*(H/w)*(W/w), w, w, C)，窗口按(batch,窗行,窗列)行主序、
    窗内保留(row,col,C)。纯重排、禁止 padding。"""
    # TODO
    raise NotImplementedError
