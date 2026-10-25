def solve(n_layers, act_bytes):
    """次线性激活检查点：peak(k) = (ceil(n/k) + k) * act_bytes, k in [1,n]。
    返回 (best_k, min_peak, store_all)：
      best_k   使 peak 最小的 k（并列取最小 k）；
      min_peak peak(best_k)（整数字节）；
      store_all n * act_bytes（全存基线，整数字节）。
    全程用整数运算（ceil 用 -(-n//k)），勿引入浮点。"""
    # TODO
    raise NotImplementedError
