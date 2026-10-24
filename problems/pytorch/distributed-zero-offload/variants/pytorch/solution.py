def solve(n_params, pcie_bandwidth, compute_time, param_bytes=2, grad_bytes=2, opt_bytes=12):
    N = float(n_params)
    gpu_mem = float(param_bytes) * N                       # 仅 fp16 参数常驻 GPU
    cpu_mem = (float(grad_bytes) + float(opt_bytes)) * N   # 梯度 + 优化器状态卸载到 CPU
    pcie_bytes = (float(grad_bytes) + float(param_bytes)) * N  # 梯度下 + 更新后参数上
    t_pcie = pcie_bytes / float(pcie_bandwidth)
    is_bottleneck = bool(t_pcie > float(compute_time))     # 严格大于
    return (float(gpu_mem), float(cpu_mem), float(pcie_bytes), is_bottleneck)
