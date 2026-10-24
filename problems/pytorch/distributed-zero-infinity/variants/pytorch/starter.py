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
    """ZeRO-Infinity 三层（HBM/DRAM/NVMe）显存-带宽账，训练态沿 dp_degree 分片。

    P = param_bytes·N/d,  G = grad_bytes·N/d,  O = opt_bytes·N/d
    梯度分片 G 恒驻 HBM；offload_param/offload_optimizer 命中的分片落入 offload_target 层。
    返回 dict:
      hbm_bytes           = G + (0 if offload_param else P) + (0 if offload_optimizer else O)
      offloaded           = (P if offload_param else 0) + (O if offload_optimizer else 0)
      dram_bytes          = offloaded if offload_target=="dram" else 0
      ssd_bytes           = offloaded if offload_target=="ssd"  else 0
      step_transfer_bytes = offloaded
    """
    # TODO: 按题面模型计算三层占用与每步搬运量
    raise NotImplementedError
