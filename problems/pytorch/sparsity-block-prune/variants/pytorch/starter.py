import torch


def solve(weight, sparsity, block):
    """块结构剪枝：每行按宽 block 切块，按块 L1(|w| 和)全局竞争，剪最小 round(sparsity*B) 个整块(置零)。
    weight float32 2-D (R,C)，C%block==0。返回同形状。并列按(行,块)展平索引升序先剪。禁止现成剪枝库。"""
    # TODO
    raise NotImplementedError
