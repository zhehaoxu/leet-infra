"""参考答案：torch.repeat_interleave 等价实现（构造下标向量 + index_select）。

逐元素重复（interleave）与整块平铺（repeat/tile）的区别只在下标向量的构造顺序：
interleave 的下标是 [0,0,1,1,2,2]，tile 的是 [0,1,2,0,1,2]。拿到下标后一切归到
index_select 这一条通路上，dim=None 只是先 reshape(-1) 再沿第 0 轴选。
repeats 为张量时逐位置次数不同，输出长度是 Σrepeats，天然支持 0 次（元素消失）。
"""

import torch


def _counts(repeats, count):
    if isinstance(repeats, torch.Tensor):
        if repeats.dim() != 1:
            raise ValueError("repeats 张量必须是一维")
        if repeats.dtype.is_floating_point or repeats.dtype.is_complex:
            raise ValueError("repeats 张量必须是整数类型")
        if repeats.numel() != count:
            raise ValueError("repeats 长度与沿 dim 轴的元素个数不符")
        r = [int(v) for v in repeats.tolist()]
    elif isinstance(repeats, int) and not isinstance(repeats, bool):
        r = [int(repeats)] * count
    else:
        raise ValueError("repeats 必须是 int 或一维整数张量")
    if any(v < 0 for v in r):
        raise ValueError("repeats 不得为负")
    return r


def solve(x, repeats, dim=None):
    if dim is None:
        y = x.reshape(-1)
        axis = 0
        count = y.numel()
    else:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError("dim 必须是 int 或 None")
        axis = dim if dim >= 0 else dim + x.dim()
        if x.dim() == 0 or not 0 <= axis < x.dim():
            raise ValueError("dim 越界")
        y = x
        count = y.shape[axis]

    r = _counts(repeats, count)
    idx = torch.tensor([i for i, k in enumerate(r) for _ in range(k)], dtype=torch.long)
    return y.index_select(axis, idx)
