import torch


def solve(x, repeats, dim=None):
    """repeat-interleave：torch.repeat_interleave 等价实现（与 repeat()/expand 的区别见下）。

    入参：
    - x: float32 张量，形状任意。
    - repeats: 非负 int，或一维整数张量（每个位置各自的重复次数，不得为负）。
    - dim: None 或 int。
      * dim=None：先把 x 展平成 1-D，再逐元素重复，返回 1-D 张量（长度 = Σrepeats）。
      * dim=int：只在 dim 轴上逐元素重复，其余维原样保留；负数按 x.dim() 折算，越界抛 ValueError。
        此时若 repeats 是张量，其长度必须等于 x.shape[dim]，否则抛 ValueError。

    语义要点（与 torch.repeat 的区别）：repeat_interleave([1,2,3], 2) = [1,1,2,2,3,3]，
    而 torch.repeat/tile 给 [1,2,3,1,2,3]。本函数实现前者：输出第 k 个位置取自
    输入沿该轴的某个原始下标，因此最省事的等价写法是先构造下标向量、再 index_select。
    repeats=0 时该位置的元素全部消失（输出长度可为 0，此时返回空张量）。
    禁止直接调用 torch.repeat_interleave。
    """
    # TODO
    raise NotImplementedError
