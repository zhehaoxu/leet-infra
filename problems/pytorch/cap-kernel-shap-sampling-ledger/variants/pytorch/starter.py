"""Captum Kernel SHAP 采样与评估次数账 —— 骨架。

你需要实现五个 mode，由 solve(mode, **kw) 分派：

  select_distribution(num_interp_features)
      → {"probs": [...], "denom": [...], "total": float,
         "normalized": [...] | None, "sampling_range": [lo, hi],
         "probs_f32": [...], "normalized_f32": [...] | None}

  similarity_weight(interpretable_sample, num_interp_features, inf_weight=1e6)
      → {"weight": float, "num_selected": int, "is_endpoint": bool}

  perturb_sample(rand_vals, num_selected_features, num_interp_features)
      → {"sample": [...], "threshold": float, "num_selected": int,
         "endpoint": bool}
  perturb_sample(kind="endpoints", num_interp_features)
      → {"samples": [[...], [...]], "num_endpoints": 2}
  perturb_sample(kind="sequence", rand_vals_list, selected_counts,
                 num_interp_features)
      → {"samples": [[...], ...], "num_endpoints": 2, "num_random": int}

  n_evaluations(total_features, n_samples, perturbations_per_eval, exact=False)
      → {"n_evaluations": int, "mode": "exact" | "sampling",
         "permutation_factor": int}

  all_permutations(num_features, num_samples=None)
      → {"permutations": [[...], ...], "count": int,
         "num_samples_ignored": bool}

关键口径：

- num_interp_features 是「可解释特征数」M，>= 1。
- interpretable_sample 是长度 M 的 0/1 向量（每个可解释特征被选中与否）。
- rand_vals 是长度 M 的浮点向量（官方是 torch.randn，这里显式入参）。
- 所有非法输入一律抛 ValueError。
- 能走精确有理数的部分走 fractions.Fraction；最终所有出参统一转 float。

六个容易写错的点：

1. p(k) 的分母是 k * (M - k)，k 的取值范围是 0..M-1（**不含 M**）。
   k=0 时分母为 0，官方用 probs[0] = 0.0 把它**手动抹掉**。
2. k = M-1 时 probs 恰为 **1.0**（分母 = (M-1)*1），**不是**无穷；
   而且 probs **不归一化**（M=5 时和为 10/3，不是 1）。
3. M=1 时 probs 全 0 ⇒ 归一化出 NaN ⇒ 官方 Categorical 抛异常，
   你的实现应让 normalized / normalized_f32 都是 None。
4. 官方全程 **float32**（torch.arange(M, dtype=torch.float)），
   M=16 时与精确有理数相差约 4.3e-08 —— probs_f32 要复现这个口径。
5. 相似度权重只在【端点】（全 0 或全 1）为 1000000.0，其余是 1.0；
   这个权重是给线性代理模型做加权最小二乘的样本权重。
6. 扰动生成的阈值取 kthvalue(rand_vals, M-k)，即**第 (M-k) 小**，
   且比较是**严格大于** —— 并列时会**少选**而不是多选。
   另外前两个样本无条件是【全 1】与【全 0】两个固定端点。

请在 solve 里完成实现，不要修改 test_cases.py。
"""

import math
from fractions import Fraction

INF_WEIGHT = 1000000.0


def _to_frac(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("unsupported numeric type: %r" % (type(value),))


def _flat(value):
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _f(value):
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


# --------------------------------------------------------------------------
# TODO: 实现下面五个函数
# --------------------------------------------------------------------------


def select_distribution(num_interp_features):
    """p(k) = (M-1)/(k*(M-k)) 的构造、probs[0]=0、归一化与 float32 口径。"""
    raise NotImplementedError("TODO: select_distribution")


def similarity_weight(interpretable_sample, num_interp_features,
                      inf_weight=INF_WEIGHT):
    """相似度权重：端点 inf_weight / 内部 1.0。"""
    raise NotImplementedError("TODO: similarity_weight")


def perturb_sample(rand_vals, num_selected_features, num_interp_features):
    """kthvalue 阈值下的严格大于 top-k 二进制向量。"""
    raise NotImplementedError("TODO: perturb_sample")


def perturb_endpoints(num_interp_features):
    """两个固定端点样本（全 1 / 全 0）。"""
    raise NotImplementedError("TODO: perturb_endpoints")


def perturb_sequence(rand_vals_list, selected_counts, num_interp_features):
    """端点 + 随机样本的完整序列。"""
    raise NotImplementedError("TODO: perturb_sequence")


def n_evaluations(total_features, n_samples, perturbations_per_eval,
                  exact=False):
    """前向评估次数（采样版 / 精确版）。"""
    raise NotImplementedError("TODO: n_evaluations")


def all_permutations(num_features, num_samples=None):
    """全部排列的枚举（num_samples 被忽略）。"""
    raise NotImplementedError("TODO: all_permutations")


# --------------------------------------------------------------------------
# 单入口
# --------------------------------------------------------------------------


def solve(mode, **kw):
    if mode == "select_distribution":
        return select_distribution(kw["num_interp_features"])
    if mode == "similarity_weight":
        return similarity_weight(
            kw["interpretable_sample"], kw["num_interp_features"],
            kw.get("inf_weight", INF_WEIGHT),
        )
    if mode == "perturb_sample":
        kind = kw.get("kind")
        if kind == "endpoints":
            return perturb_endpoints(kw["num_interp_features"])
        if kind == "sequence":
            return perturb_sequence(
                kw["rand_vals_list"], kw["selected_counts"],
                kw["num_interp_features"],
            )
        return perturb_sample(
            kw["rand_vals"], kw["num_selected_features"],
            kw["num_interp_features"],
        )
    if mode == "n_evaluations":
        return n_evaluations(
            kw["total_features"], kw["n_samples"],
            kw["perturbations_per_eval"], kw.get("exact", False),
        )
    if mode == "all_permutations":
        return all_permutations(kw["num_features"], kw.get("num_samples"))
    raise ValueError("unknown mode: %r" % (mode,))


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


if __name__ == "__main__":
    print("skeleton loaded — implement the modes above.")
