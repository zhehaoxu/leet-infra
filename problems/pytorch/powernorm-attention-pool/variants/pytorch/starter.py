import torch


def solve(x, mode="powernorm", seed=None, eps=1e-6):
    """PowerNorm 归一化 与 单种子加性注意力池化（PMA readout），mode 二选一。

    入参：
    - x: float32 / float64 张量（其他 dtype 抛 ValueError）。
      mode="powernorm" 时为 (N,C,L) 3-D 或 (N,C,H,W) 4-D；
      mode="attn-pool" 时为 (B,L,D) 3-D。维度数不符抛 ValueError。
    - mode: "powernorm" | "attn-pool"，其他取值抛 ValueError。
    - seed: 仅 attn-pool 使用——形状 (D,) 的一维查询/种子向量（Set Transformer
      PMA 的 seed vector 口径）。attn-pool 下 seed=None 抛 ValueError；
      不是一维张量或长度 != D 抛 ValueError；dtype 允许与 x 不同，会转换到
      x.dtype 后参与计算。powernorm 模式忽略该参数。
    - eps: 仅 powernorm 使用，必须为正数（否则 ValueError），默认 1e-6。
      attn-pool 模式忽略该参数。

    计算口径：
    - powernorm（PowerNorm, Yan et al. ICML 2020 的纯二阶矩口径——不做减均值、
      不除标准差）：先把 x 展平成 (N,C,M)（4-D 时 M=H*W），对每个 (n,m) 位置
      计算通道均值平方 S(n,m) = (1/C)·sum_c x[n,c,m]^2，然后
      y[n,c,m] = x[n,c,m] / sqrt(S(n,m) + eps)。eps 只进分母、防止零输入除零
      （零输入 → 0/sqrt(eps) = 0）。不做任何减均值/中心化（有专门探针）。
      返回与 x 同形状同 dtype。
    - attn-pool（单查询注意力池化，无温度参数）：logits[b,l] =
      (sum_d x[b,l,d] * q[d]) / sqrt(D)，沿 L 维（dim=-1）做数值稳定 softmax
      （减最大值再 exp 归一化）得到 attn (B,L)，输出
      out[b,d] = sum_l attn[b,l] * x[b,l,d]，形状 (B,D)。
      数学性质：所有 logits 相等（如 q 与每个 x_l 的内积相同）→ attn 均匀
      1/L，out == x.mean(dim=1)；L=1 → attn=1，out == x[:,0,:] 逐位相等。

    确定性：不使用任何随机数发生器；softmax 数值稳定（减最大值）。

    返回：powernorm → 与 x 同形状张量；attn-pool → (B,D) 张量；dtype 均同 x。
    """
    # TODO
    raise NotImplementedError
