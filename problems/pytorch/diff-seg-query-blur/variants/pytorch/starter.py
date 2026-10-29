def solve(query, blur_sigma, blur_threshold_inf=9999.0):
    """SEG 的查询平滑：方阵高斯模糊，或退化为整块均值。

    query 形如 (B, S, D) 的 query 投影（第 0 维 batch，第 1 维 token，第 2 维通道）。
    按方形图像假设，取前 n*n 个 token（n = int(sqrt(S))，向下取整）重排成 (B, D, n, n)
    的二维特征图，逐通道做平滑后写回原位；剩下的 S - n*n 个 token 原样保留。

    blur_sigma > blur_threshold_inf（严格大于）走「无限模糊」分支：
      k0 = ceil(6σ) + 1 - ceil(6σ) % 2                （保证奇数）
      k  = min(k0, n - (n % 2 - 1))                   （n 偶时上限 n+1，n 奇时上限 n）
      h  = (k - 1) / 2
      x  = linspace(-h, h, k)
      p  = exp(-0.5 * (x / blur_sigma)^2)，归一化成 1-D 高斯核
      kernel2d = outer(p, p)，按通道广播成 depthwise 卷积核 (D, 1, k, k)
      特征图四周各 pad k//2（reflect 模式）后做分组卷积
    否则走均值分支：每个 (b, d) 通道的特征图被其 n*n 个元素的均值整体替换。

    返回 float32 张量，形状与 query 相同；输入不被就地修改。
    """
    # TODO
    raise NotImplementedError
