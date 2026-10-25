"""参考答案：整数枚举 k=1..n，peak(k)=(ceil(n/k)+k)*a，升序扫描只在选择严格更小时更新（并列取最小 k）。"""


def solve(n_layers, act_bytes):
    n = int(n_layers)
    a = int(act_bytes)
    best_k = 1
    best_units = -(-n // 1) + 1  # ceil(n/1)+1 = n+1
    for k in range(2, n + 1):
        units = -(-n // k) + k  # ceil(n/k)+k，纯整数
        if units < best_units:
            best_units = units
            best_k = k
    min_peak = best_units * a
    store_all = n * a
    return (best_k, min_peak, store_all)
