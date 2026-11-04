"""Zero-Sum SVD 的全局奇异分量选择（单阶段、dense 起步）。

官方 mint-vu/Zero-Sum-SVD 的确定性核心（compression/planning.py +
utils/correction_utils.py）：

1. 白化域谱与梯度投影：A = W@L，svd(A) 得 U/Σ/Vᵀ（显式按 Σ 降序重排）；
   梯度白化 H = G@L⁻ᵀ（下三角 triangular solve），一阶损失敏感度
   grad_σ = diag(Uᵀ H V)；
2. 截断损失估计（tail 候选从最小 σ 起）：δ = −σ·grad_σ（first_order）；
   β-mixing 变体 δ = −sign(g)·|σ|^β·|g|^(1−β)；second_order 加半平方项；
3. 秩约束：r* = break_even_rank = max(1, (m·n−1)//(m+n))，
   min_rank = ceil(keep_rank_ratio·r*)，每模块最多剪 (满秩 − min_rank) 个 SV；
4. 零和选择：维护 running_sum，每轮 prefer_positive = running_sum ≤ 0，
   从对应符号堆弹 |δ| 最小的模块候选（accept 其 tail），running_sum += δ
   ——累计预测损失变化被压在 0 附近，自动产出异质秩；
5. 参数记账：dense 模块 r_now > r* 时 effective_dense_delta = 0（低秩因子
   在盈亏平衡秩之上不省参数），== r* 时 = m·n − r*·(m+n)，< r* 时 = m+n；
   剪到 stage_budget（初始参数 − (1−ratio)·初始参数）即停。

边界：多阶段/修正步（projected gradient + re-truncation）/remap 量化、
非 zero_sum 的 selection_mode 均不在判题域；模块从满秩 dense 起步。
"""

import math

import torch

_EPS = 1e-12


def _validate_2d(t, name, rows=None, cols=None):
    if not isinstance(t, torch.Tensor) or t.dim() != 2 \
            or not torch.is_floating_point(t):
        raise ValueError(f"{name} 必须是 2 维浮点张量")
    if rows is not None and t.shape[0] != rows:
        raise ValueError(f"{name} 必须是 ({rows}, {cols}) 形状")
    if cols is not None and t.shape[1] != cols:
        raise ValueError(f"{name} 必须是 ({rows}, {cols}) 形状")


def _delta_of(sigma, grad_sigma, beta_mixing, delta_order):
    if beta_mixing is None:
        if delta_order == "second_order":
            return -sigma * grad_sigma + 0.5 * sigma ** 2 * grad_sigma ** 2
        return -sigma * grad_sigma
    beta = float(beta_mixing)
    if grad_sigma == 0.0 and sigma == 0.0:
        return 0.0
    sig_abs = max(abs(sigma), _EPS)
    grad_abs = max(abs(grad_sigma), _EPS)
    sign = 1.0 if grad_sigma >= 0.0 else -1.0
    return -sign * (sig_abs ** beta) * (grad_abs ** (1.0 - beta))


def _effective_dense_delta(shape, r_now, r_star):
    m, n = shape
    if r_now > r_star:
        return 0
    if r_now == r_star:
        return m * n - r_star * (m + n)
    return m + n


def solve(modules, global_prune_ratio=0.2, keep_rank_ratio=0.3,
          beta_mixing=None, delta_order="first_order"):
    if not isinstance(modules, dict) or not modules:
        raise ValueError("modules 必须是非空 dict")
    for name, mod in modules.items():
        if not isinstance(name, str) or not name:
            raise ValueError("modules 的键必须是非空字符串")
        if not isinstance(mod, dict) or set(mod) != {"weight", "grad", "L"}:
            raise ValueError(f"modules[{name}] 必须含 weight/grad/L 三个键")
        _validate_2d(mod["weight"], f"modules[{name}].weight")
        _validate_2d(mod["grad"], f"modules[{name}].grad", *mod["weight"].shape)
        if mod["L"] is not None:
            _validate_2d(mod["L"], f"modules[{name}].L",
                         mod["weight"].shape[1], mod["weight"].shape[1])
    if isinstance(global_prune_ratio, bool) \
            or not isinstance(global_prune_ratio, (int, float)) \
            or not math.isfinite(global_prune_ratio) \
            or not (0.0 <= global_prune_ratio < 1.0):
        raise ValueError("global_prune_ratio 必须是 [0, 1) 内的有限数值")
    if isinstance(keep_rank_ratio, bool) \
            or not isinstance(keep_rank_ratio, (int, float)) \
            or not math.isfinite(keep_rank_ratio) \
            or not (0.0 <= keep_rank_ratio <= 1.0):
        raise ValueError("keep_rank_ratio 必须是 [0, 1] 内的有限数值")
    if beta_mixing is not None:
        if isinstance(beta_mixing, bool) \
                or not isinstance(beta_mixing, (int, float)) \
                or not math.isfinite(beta_mixing) \
                or not (0.0 <= beta_mixing <= 1.0):
            raise ValueError("beta_mixing 必须是 [0, 1] 内的有限数值或 None")
    if delta_order not in ("first_order", "second_order"):
        raise ValueError("delta_order 必须是 first_order 或 second_order")

    states = {}
    initial_params = 0
    for name in sorted(modules):
        mod = modules[name]
        weight, grad, L = mod["weight"], mod["grad"], mod["L"]
        m, n = weight.shape
        wd, gd = weight.double(), grad.double()
        if L is None:
            A = wd
            H = gd
        else:
            Ld = L.double()
            A = wd @ Ld
            H = torch.linalg.solve(Ld, gd.t()).t()  # G @ L^{-T}
        U, S, VT = torch.linalg.svd(A, full_matrices=False)
        order = torch.argsort(S, descending=True)
        S, U, VT = S[order], U[:, order], VT[order, :]
        V = VT.t()
        grad_sigma = torch.diagonal(U.t() @ H @ V)

        r_star = max(1, (m * n - 1) // (m + n))
        min_rank = max(1, math.ceil(keep_rank_ratio * r_star))
        max_svs = max(0, min(m, n) - min_rank)
        states[name] = {
            "shape": (m, n), "sigma": S, "grad_sigma": grad_sigma,
            "r_star": r_star, "max_svs": max_svs,
            "consumed": 0, "truncated_sq": 0.0,
        }
        initial_params += m * n

    target_params = int((1.0 - global_prune_ratio) * initial_params)
    stage_budget = initial_params - target_params

    running_sum = 0.0
    dense_removed = 0
    selections = []
    while dense_removed < stage_budget:
        prefer_positive = running_sum <= 0.0
        best = None  # (abs_delta, name, delta)
        # 官方 pop_prefer_then_any: 先偏好符号堆, 空了落到另一符号堆
        for prefer in (prefer_positive, not prefer_positive):
            for name in sorted(states):
                st = states[name]
                remaining = st["max_svs"] - st["consumed"]
                if remaining <= 0:
                    continue
                idx = st["sigma"].numel() - 1 - st["consumed"]
                delta = _delta_of(float(st["sigma"][idx]),
                                  float(st["grad_sigma"][idx]),
                                  beta_mixing, delta_order)
                if prefer and delta < 0.0:
                    continue
                if not prefer and delta >= 0.0:
                    continue
                key = (abs(delta), name)  # |δ| 最小优先; name 作确定性 tie-break
                if best is None or key < (best[0], best[1]):
                    best = (abs(delta), name, delta)
            if best is not None:
                break
        if best is None:
            break
        _, name, delta = best
        st = states[name]
        st["consumed"] += 1
        st["truncated_sq"] += float(st["sigma"][
            st["sigma"].numel() - st["consumed"]] ** 2)
        running_sum += delta
        r_now = st["sigma"].numel() - st["consumed"]
        dense_removed += _effective_dense_delta(st["shape"], r_now,
                                                st["r_star"])
        selections.append((name, delta))

    ranks, masks, truncated = {}, {}, {}
    for name in sorted(states):
        st = states[name]
        ranks[name] = st["sigma"].numel() - st["consumed"]
        masks[name] = [True] * ranks[name] + [False] * st["consumed"]
        truncated[name] = st["truncated_sq"]

    return {
        "ranks": ranks,
        "keep_masks": masks,
        "truncated_sigma_sq": truncated,
        "running_sum": running_sum,
        "dense_removed": dense_removed,
        "stage_budget": stage_budget,
        "selections": selections,
    }
