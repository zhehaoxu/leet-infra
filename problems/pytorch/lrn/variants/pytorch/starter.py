import torch


def solve(x, size, alpha=1e-4, beta=0.75, k=1.0):
    # TODO: 局部响应归一化 (LRN)。x 形状 (N, C, H, W)。
    # 对每个 (n, h, w) 沿通道维取以 i 为中心、宽 size 的窗口（边界通道补零），
    #   y_i = x_i / (k + (alpha/size) * sum_j x_j^2) ** beta
    # 注意：窗口沿 C 维滑动，H/W 不参与；平方和含自身通道；size 为奇数。
    # 结果须与 F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k) 一致。
    raise NotImplementedError
