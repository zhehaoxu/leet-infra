def solve(
    n_params: float,
    dp_degree: int,
    offload_param: bool,
    offload_optimizer: bool,
    offload_target: str,
    param_bytes: int = 2,
    grad_bytes: int = 2,
    opt_bytes: int = 12,
) -> dict:
    d = float(dp_degree)
    P = param_bytes * n_params / d
    G = grad_bytes * n_params / d
    O = opt_bytes * n_params / d

    hbm_bytes = G + (0.0 if offload_param else P) + (0.0 if offload_optimizer else O)
    offloaded = (P if offload_param else 0.0) + (O if offload_optimizer else 0.0)

    dram_bytes = offloaded if offload_target == "dram" else 0.0
    ssd_bytes = offloaded if offload_target == "ssd" else 0.0

    return {
        "hbm_bytes": float(hbm_bytes),
        "dram_bytes": float(dram_bytes),
        "ssd_bytes": float(ssd_bytes),
        "step_transfer_bytes": float(offloaded),
    }
