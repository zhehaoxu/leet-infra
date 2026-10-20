import torch


def solve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """因果滑动窗口加权平均，左侧零填充，输出与输入等长。

    x:      (B, L) 浮点序列
    kernel: (W,)，kernel[0] 对应当前时刻，kernel[i] 对应 t-i
    """
    # TODO: 1) kernel.sum()==0 时抛 ValueError
    #       2) 左侧零填充 W-1，用 unfold 造出 (B, L, W) 窗口
    #       3) 注意窗口内元素顺序与 kernel 的对齐（可能需要 flip）
    #       4) 点积后除以 kernel 全和
    # 禁止使用 F.conv1d / nn.Conv1d
    raise NotImplementedError("请实现 solve()")
