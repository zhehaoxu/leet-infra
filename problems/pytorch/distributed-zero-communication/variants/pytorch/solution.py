def solve(n_params, world, grad_bytes=2, param_bytes=2):
    N = float(n_params)
    W = float(world)
    c = (W - 1.0) / W if W > 0 else 0.0
    G = N * float(grad_bytes)   # 梯度总字节
    P = N * float(param_bytes)  # 参数总字节
    ddp = 2.0 * c * G
    zero_1_2 = 2.0 * c * G                 # reduce-scatter 梯度 + all-gather 更新后参数
    zero_3 = 2.0 * c * G + 2.0 * c * P     # 再叠加前向/反向各一次 all-gather 参数
    return (float(ddp), float(zero_1_2), float(zero_3))
