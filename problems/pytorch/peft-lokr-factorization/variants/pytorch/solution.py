def solve(dimension, factor):
    """LoKr factorization：factor>0 且整除时返回 (factor, dimension//factor) 原序；
    factor==-1 时返回 m+n 最小（m 最接近 sqrt）的因子对 (m, n)，m<=n。
    非法输入抛 ValueError。返回 (m, n) 两个 int。"""
    dimension, factor = int(dimension), int(factor)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if factor > 0:
        if dimension % factor != 0:
            raise ValueError("factor must divide dimension")
        return (factor, dimension // factor)
    if factor != -1:
        raise ValueError("factor must be -1 or a positive divisor")
    best_m, best_n = 1, dimension
    best_sum = 1 + dimension
    m = 1
    while m * 2 <= dimension:
        if dimension % m == 0:
            n = dimension // m
            if m + n < best_sum:
                best_sum = m + n
                best_m, best_n = m, n
        m += 1
    return (best_m, best_n)
