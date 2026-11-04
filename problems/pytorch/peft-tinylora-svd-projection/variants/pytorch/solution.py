"""TinyLoRA 的 SVD 分摊 + 随机投影账。

回源: HF PEFT src/peft/tuners/tinylora/{layer,model}.py（PR #3024 已合并；
论文 Learning to Reason in 13 Parameters, arXiv:2602.04118，Meta；参考实现
ANRGUSC/TinyLoRA）。机制：LoRA-XS 骨架（冻结 SVD 因子 + 中间小矩阵 R）
再进一步——R 参数化为固定随机投影矩阵的线性组合 R = Σ_i v[i]·P[i]，
唯一可训练参数是长度 u 的向量 v（论文里小到 13 个 bf16 值）。

三处确定性账:
- SVD 分摊（_init_svd）: W = U·diag(S)·Vh，**√S 均分两侧**——
  tinylora_A = diag(√S[:r])·Vh[:r,:] (r, in)、
  tinylora_B = U[:,:r]·diag(√S[:r]) (out, r)；
  actual_r = min(r, min(out, in))（短边截断）。两侧范数均衡，
  ΔW = B @ R @ A。
- 投影初始化（_init_projection）: P (u, r, r) ~ N(0, 1/r)
  （std = 1/√r，Johnson-Lindenstrauss 惯例），seed = base_seed +
  layer_idx 逐层确定（torch.Generator().manual_seed）；P 冻结
  （BufferDict，非参数）。
- R 组合（_compute_R）: einsum("i,ijk->jk", v, P)——v[i] 加权 P[i]。
  v 零初始化（init_weights=True 默认）=> 训练起点 ΔW = 0。

weight_tying 分组（model.py）: 目标层按注入顺序拿确定层号 idx，
num_groups = max(1, round(n·(1-wt)))（wt=0 不共享、wt=1 全共享）、
group_size = max(1, n // num_groups)、group_idx = min(idx // group_size,
num_groups - 1)——同组层经模型级 ModuleDict 共享同一个 v。

前向 delta = x @ Aᵀ @ Rᵀ @ Bᵀ（三次 F.linear：A→R→B），
merge/unmerge = W ± B@R@A（safe_merge 加有限性检查）。
六入口复刻（SVD 因子 U/S/Vh 作显式入参，离线算好）:
- svd_split / init_projection / compute_R / delta_weight /
  forward_delta / tie_groups
"""

import torch


def svd_split(U, S, Vh, r):
    """官方 _init_svd 的分摊账：√S 均分两侧 + 短边截断。"""
    max_rank = min(U.shape[0], Vh.shape[1])
    actual_r = min(r, max_rank)
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    V_r = Vh[:actual_r, :]
    sqrt_S = torch.sqrt(S_r)
    A = (sqrt_S.unsqueeze(1) * V_r).contiguous()   # (r, in)
    B = (U_r * sqrt_S.unsqueeze(0)).contiguous()   # (out, r)
    return A, B, actual_r


def init_projection(u, r, seed):
    """官方 _init_projection：P (u, r, r) ~ N(0, 1/r)，逐层确定 seed。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.normal(mean=0.0, std=1.0 / (r ** 0.5), size=(u, r, r),
                        generator=gen)


def compute_R(v, P):
    """官方 _compute_R：R = Σ_i v[i]·P[i]，einsum("i,ijk->jk")。"""
    return torch.einsum("i,ijk->jk", v, P)


def delta_weight(tinylora_B, R, tinylora_A):
    """官方 get_delta_weight：ΔW = B @ R @ A，(out, r) @ (r, r) @ (r, in)。"""
    return tinylora_B @ R @ tinylora_A


def forward_delta(x, tinylora_A, R, tinylora_B):
    """官方 Linear.forward 的 delta 路径：x @ Aᵀ @ Rᵀ @ Bᵀ。

    LoRA-XS 模式三次 F.linear：F.linear(x, A) -> (…, r)、
    F.linear(h, R) -> (…, r)、F.linear(h, B) -> (…, out)。
    """
    h = x @ tinylora_A.T
    h = h @ R.T
    return h @ tinylora_B.T


def tie_groups(num_layers, weight_tying):
    """官方 model.py 的分组账：确定层号 -> v 共享组。"""
    num_groups = max(1, round(num_layers * (1.0 - weight_tying)))
    group_size = max(1, num_layers // num_groups)
    indices = [min(i // group_size, num_groups - 1) for i in range(num_layers)]
    return {"num_groups": num_groups, "group_size": group_size,
            "group_indices": indices}


def solve(mode=None, **kw):
    if mode == "svd_split":
        return svd_split(kw["U"], kw["S"], kw["Vh"], kw["r"])
    if mode == "init_projection":
        return init_projection(kw["u"], kw["r"], kw["seed"])
    if mode == "compute_R":
        return compute_R(kw["v"], kw["P"])
    if mode == "delta_weight":
        return delta_weight(kw["tinylora_B"], kw["R"], kw["tinylora_A"])
    if mode == "forward_delta":
        return forward_delta(kw["x"], kw["tinylora_A"], kw["R"],
                             kw["tinylora_B"])
    if mode == "tie_groups":
        return tie_groups(kw["num_layers"], kw["weight_tying"])
    raise ValueError("unknown mode: %s" % mode)
