import torch


def solve(x, expert_ids, routing_weights, gate_w, up_w, down_w,
          shared_gate_w, shared_up_w, shared_down_w):
    """中文契约 / English contract
    x [T,D], expert_ids [T,K] int64, routing_weights [T,K] 不重新归一化 / not renormalized.
    gate_w/up_w [E,H,D], down_w [E,D,H]；shared_* 的形状为 [Hs,D], [Hs,D], [D,Hs] /
        routed weights [E,H,D], [E,H,D], [E,D,H]; shared shapes [Hs,D], [Hs,D], [D,Hs].
    CPU float64，支持非连续张量且不修改输入；无偏置、残差、dropout 或归一化 /
        CPU float64; accept noncontiguous tensors and mutate nothing; no bias/residual/dropout/norm.
    共享 SwiGLU 常开且无门控，输出为 routed + shared /
        the shared SwiGLU is always on and ungated; output is routed + shared.
    返回 / Returns (output [T,D], routed [T,D], shared [T,D], total_params int, active_params int)；
        active_params 为每 token 所选路径的结构参数量，不是 batch 并集或乘 token 数 /
        active_params is the per-token selected-path structural count, not a batch union or token multiple.
    """
    raise NotImplementedError
