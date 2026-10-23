import torch


def solve(x, weight, bias, groups):
    # TODO: 分组卷积。weight 形状 (Cout, Cin//groups, kh, kw)，stride=1、padding=0。
    # 第 g 组：输入通道 [g*(Cin//groups):(g+1)*(Cin//groups)] 只与该组对应的
    # 输出通道 [g*(Cout//groups):(g+1)*(Cout//groups)] 相连（组间隔离）。
    raise NotImplementedError
