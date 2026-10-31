"""参考解：ShiftAddLLM 的贪心位平面 + ApOT（加性 2 的幂）分解。

逐步转写自 GATECH-EIC/ShiftAddLLM 的 quantizers/bcq_quant/bcq_shift.py：
  get_shift_and_sign    L15  -> _shift_and_sign（rounding 取默认 deterministic）
  round_power_of_2      L26  -> _round_power_of_2（q_bias/scale 恒为 None）
  additive_power_of_2   L41  -> _additive_power_of_2（内部硬编码 deterministic）
  greedy_mean_torch     L131 -> 位平面贪心拟合，wf=ones、shift=False 分支
  quantize_shift L106-107    -> group_size<=0 退化成整行；view([-1, group_size])
  quantize_shift L115        -> einsum('ijl,il->ij', (B, alpha))
"""

import torch

LN2 = 0.6931471805599453  # np.log(2)：上游 get_shift_and_sign 的除数


def _shift_and_sign(x):
    """bcq_shift.py:15 —— sign=sign(x)，shift=round(log|x| / ln2)；x==0 时 log 给 -inf。"""
    sign = torch.sign(x)
    shift = torch.round(torch.log(torch.abs(x)) / LN2)
    return shift, sign


def _round_power_of_2(x):
    """bcq_shift.py:26 —— q_bias/scale 均为 None，只剩 2**shift * sign。"""
    shift, sign = _shift_and_sign(x)
    return (2.0 ** shift) * sign


def _additive_power_of_2(x, nums):
    """bcq_shift.py:41 —— 首项取 x 自身的最近幂，之后每次对"残差 x - 已提交和"再取一项。

    上游只返回累加和；这里额外快照每一项，便于判题定位是哪一步走偏。
    """
    total = _round_power_of_2(x)
    terms = torch.zeros(x.shape[0], x.shape[1], nums, dtype=x.dtype)
    terms[:, :, 0] = total
    for j in range(1, nums):
        shift, sign = _shift_and_sign(x - total)
        term = (2.0 ** shift) * sign
        total = total + term
        terms[:, :, j] = term
    return total, terms


def solve(weight, num_bits, group_size=-1, apot_nums=1):
    if not isinstance(weight, torch.Tensor):
        raise ValueError("weight must be a torch.Tensor")
    if weight.dim() != 2:
        raise ValueError("weight must be a 2-D tensor")
    if weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("weight must be float32 or float64")
    if weight.numel() == 0:
        raise ValueError("empty weight is not supported")
    for name, value in (("num_bits", num_bits), ("group_size", group_size),
                        ("apot_nums", apot_nums)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int")
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if apot_nums < 1:
        raise ValueError("apot_nums must be >= 1")

    rows, n_cols = int(weight.shape[0]), int(weight.shape[1])
    g = n_cols if group_size <= 0 else int(group_size)
    if g <= 0 or rows * n_cols % g != 0:
        raise ValueError("group_size must divide the flattened element count")

    work = torch.float64
    w64 = weight.detach().to(work)
    out_dtype = weight.dtype

    groups = w64.reshape(-1, g)  # bcq_shift.py:107 —— 行主序展平后切组，组可以跨行
    n_groups = int(groups.shape[0])

    # bcq_shift.py:131 greedy_mean_torch，wf=ones（分母恒为 g）、shift=False
    r = groups.clone()
    planes = torch.zeros(n_groups, g, num_bits, dtype=work)
    alpha = torch.zeros(n_groups, num_bits, dtype=work)
    for i in range(num_bits):
        b = torch.sign(r)
        a = (r.abs() * torch.ones_like(r)).sum(dim=1) / float(g)
        a = torch.where(torch.isnan(a), torch.zeros_like(a), a)  # wf=ones 时不会触发
        r = r - b * a.unsqueeze(1)
        planes[:, :, i] = b
        alpha[:, i] = a

    alpha_apot, alpha_terms = _additive_power_of_2(alpha, int(apot_nums))

    rec = torch.einsum("ijl,il->ij", planes, alpha_apot)  # bcq_shift.py:115
    reconstructed = rec.reshape(rows, n_cols)
    mse = ((w64 - reconstructed) ** 2).mean()

    return {
        "group_shape": (n_groups, g),
        "planes": planes.to(out_dtype),
        "alpha": alpha.to(out_dtype),
        "alpha_apot": alpha_apot.to(out_dtype),
        "apot_terms": alpha_terms.to(out_dtype),
        "recon": reconstructed.to(out_dtype),
        "mse": mse.to(out_dtype),
    }
