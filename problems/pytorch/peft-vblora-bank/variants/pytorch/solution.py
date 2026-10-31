"""参考答案：VB-LoRA 向量库选择适配器。"""

import torch


def _float_tensor(name, value, ndim_min, ndim_max=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} 必须是 float32/float64 张量")
    if value.dim() < ndim_min:
        raise ValueError(f"{name} 维度不足")
    if ndim_max is not None and value.dim() > ndim_max:
        raise ValueError(f"{name} 维度过高")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN/Inf")
    return value


def _select(logits, bank, topk):
    """每个 logit 向量选 topk 个库向量并加权求和；softmax 只覆盖被选中的 topk 个。

    并列值按下标升序取（稳定排序），与 `torch.topk` 的边界并列行为可能不同，题面已钉死该口径。
    """
    indices = torch.argsort(-logits, dim=-1, stable=True)[..., :topk]
    top_logits = torch.gather(logits, -1, indices)
    weights = torch.softmax(top_logits, dim=-1)
    return (weights.unsqueeze(-1) * bank[indices]).sum(-2)


def solve(x, W0, logits_A, logits_B, vector_bank, b0=None, topk=2):
    x = _float_tensor("x", x, 1)
    W0 = _float_tensor("W0", W0, 2, 2)
    logits_A = _float_tensor("logits_A", logits_A, 3, 3)
    logits_B = _float_tensor("logits_B", logits_B, 3, 3)
    vector_bank = _float_tensor("vector_bank", vector_bank, 2, 2)

    if isinstance(topk, bool) or not isinstance(topk, int):
        raise ValueError("topk 必须是整数")

    out_dim, in_dim = W0.shape
    if in_dim == 0 or out_dim == 0:
        raise ValueError("W0 不能有空维度")
    if x.shape[-1] != in_dim:
        raise ValueError("x 的最后一维必须等于 d_in")

    num_vectors, vector_length = vector_bank.shape
    if num_vectors <= 0 or vector_length <= 0:
        raise ValueError("vector_bank 不能为空")
    if topk <= 0:
        raise ValueError("topk 必须为正")
    if topk > num_vectors:
        raise ValueError("topk 不能超过 num_vectors")
    if in_dim % vector_length != 0 or out_dim % vector_length != 0:
        raise ValueError("d_in 与 d_out 都必须能被 vector_length 整除")

    rank = logits_A.shape[0]
    if rank <= 0:
        raise ValueError("r 必须为正")
    if logits_A.shape != (rank, in_dim // vector_length, num_vectors):
        raise ValueError("logits_A 形状必须是 (r, d_in//vector_length, num_vectors)")
    if logits_B.shape != (out_dim // vector_length, rank, num_vectors):
        raise ValueError("logits_B 形状必须是 (d_out//vector_length, r, num_vectors)")

    if b0 is not None:
        b0 = _float_tensor("b0", b0, 1, 1)
        if b0.numel() != out_dim:
            raise ValueError("b0 长度必须等于 d_out")

    x64 = x.to(torch.float64)
    weight = W0.to(torch.float64)
    bank = vector_bank.to(torch.float64)
    la = logits_A.to(torch.float64)
    lb = logits_B.to(torch.float64)

    A = _select(la, bank, topk).reshape(rank, -1)
    B = _select(lb, bank, topk).transpose(1, 2).reshape(-1, rank)
    delta = B @ A

    y = x64 @ weight.T
    if b0 is not None:
        y = y + b0.to(torch.float64)
    y = y + x64 @ delta.T
    return (
        y.to(torch.float32),
        A.to(torch.float32),
        B.to(torch.float32),
        delta.to(torch.float32),
    )
