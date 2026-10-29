import torch


def solve(V, g, dW):
    """weight_norm 重参数化的反向：W = g · V / ‖V‖（按第 0 轴切片），返回 (dg, dV)。

    前向：对每个输出切片 i，w_i = g_i · v_i / ‖v_i‖₂，其中 ‖v_i‖ 把该切片的全部元素
    折成一根数（V 可以是 (out, in) 的线性权重，也可以是 (out, in, kh, kw) 的卷积权重）。

    入参：
    - V: 至少二维，第 0 轴是输出通道；g: (out,)；dW: 与 V 同形（对等价权重 W 的梯度）。
    - 形状不符、V 少于一维、‖v_i‖ = 0（零向量切片，方向无定义）都抛 ValueError。

    口径：
    - dg_i = Σ_j dW_{i,j} · v̂_{i,j}，其中 v̂ = v/‖v‖（即 dW 在单位方向上的投影）。
    - dV_i = (g_i / ‖v_i‖) · (dW_i − v̂_i · dg_i)：**必须先把 dW 沿 v̂ 的分量减掉**，
      否则与 dg 重复计入同一个自由度（weight_norm 的重参数化对 v 的尺度不可辨识）。
    - 由此得到两条强自检：① Σ_j dV_{i,j}·v_{i,j} = 0（dV 与 v 正交）；
      ② 把 V 的第 i 片整体缩放 c 倍（g、dW 不变）时 W 与 dg 都不变。
    - 内部 float64 计算，dg 的 dtype 与 g 一致、dV 与 V 一致。
    """
    # TODO
    raise NotImplementedError
