import torch


def solve(inputs: list[torch.Tensor], root: int) -> list[torch.Tensor]:
    """Reduce：所有 rank 逐元素求和，结果只放在 root rank，其余 rank 保持原值。

    inputs : 长度 W 的 list，inputs[r] 是 rank r 的张量（形状全部一致）
    root   : 汇聚目标 rank 下标（0 <= root < W）
    返回   : 长度 W 的 list；[root] == Σ inputs，其余 == 各自原始输入（不广播、不清零）
    不得修改输入张量。
    """
    # TODO: 求全局和，仅写入 root 项，其余原样返回
    raise NotImplementedError
