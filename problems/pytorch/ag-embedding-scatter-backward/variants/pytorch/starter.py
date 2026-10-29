import torch


def solve(grad_out, idx, num_embeddings, padding_idx=None, scale_grad_by_freq=False, sparse=False):
    """F.embedding 的反向：把 grad_out 按 idx 散点累加成词表梯度，返回 (num_embeddings, dim)。

    前向：out[i] = weight[idx[i]]，weight 形状 (num_embeddings, dim)。

    入参：
    - grad_out: (n, dim) 二维上游梯度；idx: (n,) 一维 int64，取值必须在 [0, num_embeddings)。
    - num_embeddings: int，词表行数（决定输出形状）。
    - padding_idx: int 或 None；该行梯度必须整行置 0（前向它是常量，不该收到梯度）。
    - scale_grad_by_freq: True 时每个 token 的梯度先除以该 id 在 idx 中的出现次数
      （即「按词频摊薄」，与 F.embedding 的语义一致），再累加到同一行。
    - sparse: True 时返回 **已合并（coalesced）的 torch.sparse_coo_tensor**，
      其 to_dense() 必须与稠密路径逐位一致；False 时返回普通稠密张量。

    要求：
    - 重复 id 是**累加**而不是覆盖：idx=[1,1,0] 时第 1 行是两行梯度之和。
    - 未被任何 token 命中的行全为 0。
    - idx 越界、grad_out 非二维、idx 与 grad_out 行数不符、num_embeddings <= 0
      都抛 ValueError。dtype 与 grad_out 一致（内部 float64 累计）。
    """
    # TODO
    raise NotImplementedError
