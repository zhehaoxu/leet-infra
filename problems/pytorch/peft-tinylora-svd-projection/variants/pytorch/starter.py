"""TinyLoRA (official ANRGUSC/TinyLoRA reference pack, merged into HF PEFT
tuners/tinylora via PR #3024; paper "Learning to Reason in 13 Parameters",
arXiv:2602.04118) -- frozen SVD factors + random projection ledger.

Dispatch by the ``mode`` keyword:

- mode="svd_split": the _init_svd split account (kwargs: U, S, Vh, r)
  -> (tinylora_A, tinylora_B, actual_r); sqrt(S) is split EQUALLY to
  both sides -- A = diag(sqrt(S[:r])) @ Vh[:r,:], B = U[:,:r] @
  diag(sqrt(S[:r])); actual_r = min(r, min(out, in)) clamps the rank
- mode="init_projection": the fixed random projections
  (kwargs: u, r, seed) -> P (u, r, r) ~ N(0, 1/r) drawn from a
  torch.Generator seeded with `seed` (= base_seed + layer_idx);
  P is FROZEN (a buffer, not a parameter)
- mode="compute_R": R = sum_i v[i] * P[i] via einsum("i,ijk->jk")
  (kwargs: v (u,), P (u, r, r)) -> (r, r); v is the ONLY trainable
  tensor, zero-initialized by default
- mode="delta_weight": dW = B @ R @ A (kwargs: tinylora_B, R,
  tinylora_A) -> (out, in)
- mode="forward_delta": the LoRA-XS chain x @ A^T @ R^T @ B^T
  (kwargs: x (…, in), tinylora_A, R, tinylora_B) -> (…, out)
- mode="tie_groups": the weight_tying ledger
  (kwargs: num_layers, weight_tying) -> {"num_groups",
  "group_size", "group_indices"} with num_groups = max(1,
  round(n * (1 - wt))), group_size = max(1, n // num_groups),
  group_idx = min(idx // group_size, num_groups - 1); layers in the
  same group share one trainable v

Dropout is fixed at 0; the SVD factors U/S/Vh are explicit inputs
(computed offline), and only the deterministic projection accounts
are replicated.
"""

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
    raise NotImplementedError


def init_projection(u, r, seed):
    raise NotImplementedError


def compute_R(v, P):
    raise NotImplementedError


def delta_weight(tinylora_B, R, tinylora_A):
    raise NotImplementedError


def forward_delta(x, tinylora_A, R, tinylora_B):
    raise NotImplementedError


def tie_groups(num_layers, weight_tying):
    raise NotImplementedError


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
