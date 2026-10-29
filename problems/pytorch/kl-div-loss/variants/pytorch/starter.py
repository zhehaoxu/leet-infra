import torch


def solve(log_pred, target, reduction="batchmean", log_target=False):
    """kl-div-loss：F.kl_div 等价实现——注意第一个参数是 log q（对数概率），不是 q。

    入参：
    - log_pred: float32 张量 input，值为 log q（例如 log_softmax 的输出），形状 (B, ...)。
    - target: float32 张量 p（或 log p，当 log_target=True），可与 log_pred 广播。
    - log_target: False（默认）时 target 是概率 p，逐元素为 p·(log p − log q)；
      True 时 target 已经是 log p，逐元素为 exp(target)·(target − log_pred)。
    - reduction: "batchmean"（默认）/ "mean" / "sum" / "none"：
      * "none" 返回逐元素张量（广播后的形状）；
      * "sum" 是逐元素之和；
      * "mean" 是逐元素之和 ÷ 元素个数；
      * "batchmean" 是逐元素之和 ÷ 输出的第 0 维长度 B——KL(P‖Q) 的数学定义要求这个口径，
        而 PyTorch 的默认值 "mean" 反而少除了一个类数，这是蒸馏代码里最常见的隐藏 bug。
      逐元素口径与 F.kl_div 一致：p > 0 时为 p·(log p − log q)，p > 0 且 q = 0 时为 +inf。
      p = 0 处本题统一采用数学约定 0·log 0 := 0（贡献恰为 0），不复制 PyTorch 在
      p = 0 且 q = 0 时返回 NaN 的实现细节；因此除该退化格外，稠密（p 全 > 0）时逐点等于 F.kl_div。
      reduction 非法时抛 ValueError。

    内部 float64 计算，返回 float32。与已有的 kl-divergence 题不同：那题交的是
    两个概率分布之间的标量 KL(P‖Q)/KL(Q‖P)，本题交的是 PyTorch 损失函数的接口口径。
    """
    # TODO
    raise NotImplementedError
