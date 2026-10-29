import torch


def solve(x, n=1, dim=-1, prepend=None, append=None):
    """tensor-diff：torch.diff 等价实现，沿一根轴做 n 阶相邻差分。

    入参：
    - x: float32 张量，形状任意（至少一维）。
    - n: 非负 int，默认 1。n 阶差分就是对差分结果再做 n−1 次一阶差分；n=0 返回与输入
      内容相同的副本（与 torch 口径一致）；负数或非 int 抛 ValueError。
    - dim: int，默认 -1（最后一维）；负数按 x.dim() 折算，越界抛 ValueError。
    - prepend / append: None 或与 x 同 ndim、除 dim 轴外其余维完全一致的 float32 张量。
      它们分别在 dim 轴的前后拼接进来再参与差分：prepend 的最后一行与 x 的第一行相邻，
      x 的最后一行与 append 的第一行相邻。形状不符抛 ValueError。

    输出沿 dim 轴的长度 = x.shape[dim] + prepend.shape[dim] + append.shape[dim] − n
    （prepend/append 为 None 时按 0 计）。若结果长度 ≤ 0，抛 ValueError（torch 给空张量，
    本题统一要求非空以便判题）。其余维与输入一致，dtype 与输入相同。
    """
    # TODO
    raise NotImplementedError
