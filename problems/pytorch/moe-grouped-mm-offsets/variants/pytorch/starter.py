import torch


def solve(x, weight, offs, grad_output):
    """中文契约 / English contract
    x [S,K] 与 grad_output [S,N] 是 CPU float64、非空 2-D、有限；weight [E,K,N] 是 CPU float64、
    非空 3-D、有限；offs [E] 是 CPU int32 或 int64、非空 1-D /
        x and grad_output are CPU float64 non-empty 2-D finite; weight is CPU float64 non-empty 3-D
        finite; offs is a CPU int32/int64 non-empty 1-D tensor.
    维度必须对齐：x.size(1)==weight.size(1)、weight.size(2)==grad_output.size(1)、
    grad_output.size(0)==x.size(0)、offs.size(0)==weight.size(0)。
    offs 是每组**结束处**的行下标（前缀和，左含右开）：非降、元素 >= 0、offs[E-1] <= S /
        offs[e] is the exclusive end row of group e: non-decreasing, >= 0, offs[-1] <= S.
    第 e 组占行 [start_e, end_e) = [offs[e-1] if e else 0, offs[e])；
    out[start:end] = x[start:end] @ weight[e]；
    grad_input[start:end] = grad_output[start:end] @ weight[e].T；
    grad_weight[e] = x[start:end].T @ grad_output[start:end]；counts[e] = end - start。
    空组（start == end）整段跳过，grad_weight[e] 全 0；[offs[E-1], S) 的尾部行不属于任何专家，
    out 与 grad_input 的对应行必须是 0，且不贡献任何 grad_weight /
        empty groups are skipped; rows past offs[E-1] belong to no expert and stay zero.
    非法输入（dtype/形状/设备/非有限/维度不匹配/offs 降序或越界）抛 ValueError；
    不修改输入，接受非连续张量 / raise ValueError otherwise; mutate nothing; accept noncontiguous inputs.
    返回 / Returns (out float64 [S,N], grad_input float64 [S,K], grad_weight float64 [E,K,N],
        counts int64 [E])。
    """
    raise NotImplementedError
