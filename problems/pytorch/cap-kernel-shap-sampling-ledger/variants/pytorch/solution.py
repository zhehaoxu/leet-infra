"""Captum Kernel SHAP 采样与评估次数账（纯 Python）。

对应官方源码（pytorch/captum, main 分支）：
  captum/attr/_core/kernel_shap.py::KernelShap.attribute
      —— probs 的构造（arange / denom / probs[0]=0）与 Categorical 采样分布
  captum/attr/_core/kernel_shap.py::kernel_shap_similarity_kernel
      —— 端点 inf_weight 与内部 1.0 的相似度权重
  captum/attr/_core/kernel_shap.py::kernel_shap_perturb_generator
      —— 两个固定端点样本 + kthvalue 阈值 + 严格大于的 top-k 选择
  captum/attr/_core/shapley_value.py::ShapleyValueSampling._get_n_evaluations
  captum/attr/_core/shapley_value.py::ShapleyValues._get_n_evaluations
      —— 采样版与精确版的前向评估次数口径

四个 mode（全部显式接收长度，不依赖任何隐式状态）：
  select_distribution  —— p(k) = (M-1)/(k*(M-k)) 的构造与归一化
  similarity_weight   —— 相似度权重（端点 inf_weight / 内部 1.0）
  perturb_sample      —— 二进制扰动向量的生成（端点 + top-k 阈值选择）
  n_evaluations       —— 前向评估次数（采样版 ceil * n_samples / 精确版 factorial）

数值口径：能走精确有理数（Fraction）的部分一律走 Fraction；
最终所有出参统一转成 float。
"""

import math
from fractions import Fraction


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

_EPS = 1e-12

# 官方 KernelShap.__init__ 里写死的端点权重
INF_WEIGHT = 1000000.0


def _to_frac(value):
    """把输入转成精确有理数 Fraction。"""
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
    """把嵌套列表展平成一维列表。"""
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _f(value):
    """出参统一转 float。"""
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


def _check_pos_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an int, got %r" % (name, type(value)))
    if value < 1:
        raise ValueError("%s must be >= 1, got %d" % (name, value))
    return value


# --------------------------------------------------------------------------
# mode 1: select_distribution
# --------------------------------------------------------------------------


def select_distribution(num_interp_features):
    """官方 p(k) 的构造与归一化。

    官方实现（captum/attr/_core/kernel_shap.py::KernelShap.attribute）：

        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        ... num_select_distribution=Categorical(probs)

    ★ 要点：
      1. `num_features_list = arange(M)` ⇒ k 的取值范围是 **0 .. M-1**
         （**不含 M**）。
      2. `denom = k * (M - k)`。k=0 时 denom=0 ⇒ `(M-1)/0 = inf`，
         官方用 `probs[0] = 0.0` 把它**手动抹掉**。
      3. ★ k=M-1 时 denom = (M-1)*1 = M-1 ⇒ probs = **(M-1)/(M-1) = 1.0**，
         **不是**无穷。也就是说「全选」这个端点的概率权重是 1，
         而不是理论上应该无穷大。
      4. ★ probs **不归一化**：M=5 时 probs 之和是 10/3 而不是 1。
         `Categorical(probs)` 会在内部归一化，所以真正生效的是
         `probs / probs.sum()`。
      5. ★ M=1 时 probs = [0.0]，和为 0 ⇒ 归一化出 NaN ⇒
         `Categorical` 抛 `ValueError`（Simplex 约束失败）。
         **官方 Kernel SHAP 在只有一个可解释特征时直接崩。**
      6. ★ 官方用 **float32**（`torch.arange(M, dtype=torch.float)`）
         算 probs，所以官方数值带 float32 精度误差（M=16 时与精确值
         相差约 4.3e-08）。本 mode 同时给出精确有理数口径（`probs` /
         `total` / `normalized`）与官方 float32 口径（`probs_f32` /
         `normalized_f32`）。

    返回：
      {"probs": [...], "denom": [...], "total": float,
       "normalized": [...] | None, "sampling_range": [lo, hi],
       "probs_f32": [...], "normalized_f32": [...] | None}
    """
    M = _check_pos_int(num_interp_features, "num_interp_features")

    # denom = k * (M - k)，k 取 0..M-1
    denom = [Fraction(k * (M - k)) for k in range(M)]

    # probs = (M - 1) / denom，然后 probs[0] = 0.0
    probs = []
    for k in range(M):
        d = denom[k]
        if d == 0:
            # k=0：官方 (M-1)/0 = inf，然后被 probs[0]=0.0 覆盖
            probs.append(Fraction(0))
        else:
            probs.append(Fraction(M - 1) / d)
    probs[0] = Fraction(0)

    total = Fraction(0)
    for p in probs:
        total += p

    normalized = None
    if total != 0:
        normalized = [p / total for p in probs]

    # 采样得到的 k 的可能范围：probs > 0 的下标
    positive = [k for k, p in enumerate(probs) if p > 0]
    if positive:
        sampling_range = [positive[0], positive[-1]]
    else:
        sampling_range = [0, -1]   # 空区间（M=1 的退化情形）

    # ★ 官方 float32 口径：torch.arange(M, dtype=torch.float) 全程 float32
    probs_f32 = _float32_probs(M)
    total_f32 = 0.0
    for v in probs_f32:
        total_f32 += v
    normalized_f32 = None
    if total_f32 != 0.0:
        normalized_f32 = [v / total_f32 for v in probs_f32]

    return {
        "probs": [_f(p) for p in probs],
        "denom": [_f(d) for d in denom],
        "total": _f(total),
        "normalized": None if normalized is None else [_f(p) for p in normalized],
        "sampling_range": sampling_range,
        "probs_f32": probs_f32,
        "normalized_f32": normalized_f32,
    }


def _float32_probs(M):
    """官方 float32 口径的 probs（逐步模拟 float32 算术）。

    官方：
        num_features_list = torch.arange(M, dtype=torch.float)
        denom = num_features_list * (M - num_features_list)
        probs = (M - 1) / denom
        probs[0] = 0.0
    """
    import struct

    def f32(x):
        return struct.unpack("f", struct.pack("f", float(x)))[0]

    nf = [f32(k) for k in range(M)]
    denom = [f32(f32(nf[k]) * f32(f32(M) - nf[k])) for k in range(M)]
    probs = []
    for k in range(M):
        if denom[k] == 0.0:
            probs.append(float("inf"))
        else:
            probs.append(f32(f32(M - 1) / denom[k]))
    probs[0] = 0.0
    return probs


# --------------------------------------------------------------------------
# mode 2: similarity_weight
# --------------------------------------------------------------------------


def similarity_weight(interpretable_sample, num_interp_features,
                      inf_weight=INF_WEIGHT):
    """官方 kernel_shap_similarity_kernel 的权重口径。

    官方实现（captum/attr/_core/kernel_shap.py::kernel_shap_similarity_kernel）：

        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])

    ★ 要点：
      1. ★ 端点（**全 0** 或 **全 1**）的权重是 `inf_weight = 1000000.0`，
         其余样本权重一律是 `1.0`。
      2. ★ 这个权重是给线性代理模型做**加权最小二乘**的样本权重。
         官方注释说得很清楚：端点的理论权重应该是无穷大，
         「实践中只要让这个权重足够大（1000000）就行，
         这样能强制训练出的线性模型满足端点准则」。
      3. 判据是「选中的特征数」，**不是**样本里 1 的位置。
      4. `interpretable_sample` 是 0/1 向量；长度必须等于
         `num_interp_features`，否则官方会因 broadcast 出错。

    返回：
      {"weight": float, "num_selected": int, "is_endpoint": bool}
    """
    M = _check_pos_int(num_interp_features, "num_interp_features")
    if inf_weight is None:
        inf_weight = INF_WEIGHT
    inf_weight = _to_frac(inf_weight)

    sample = [int(v) for v in _flat(interpretable_sample)]
    for v in sample:
        if v not in (0, 1):
            raise ValueError(
                "interpretable_sample must be a 0/1 vector, got %r" % (v,))
    if len(sample) != M:
        raise ValueError(
            "interpretable_sample length %d must equal num_interp_features %d"
            % (len(sample), M))

    num_selected = sum(sample)
    is_endpoint = (num_selected == 0 or num_selected == M)
    weight = inf_weight if is_endpoint else Fraction(1)

    return {
        "weight": _f(weight),
        "num_selected": num_selected,
        "is_endpoint": is_endpoint,
    }


# --------------------------------------------------------------------------
# mode 3: perturb_sample
# --------------------------------------------------------------------------


def perturb_sample(rand_vals, num_selected_features, num_interp_features):
    """官方 kernel_shap_perturb_generator 的二进制向量生成。

    官方实现（captum/attr/_core/kernel_shap.py::kernel_shap_perturb_generator）：

        yield torch.ones(1, num_features, device=device, dtype=torch.long)
        yield torch.zeros(1, num_features, device=device, dtype=torch.long)
        while True:
            num_selected_features = kwargs["num_select_distribution"].sample()
            rand_vals = torch.randn(1, num_features)
            threshold = torch.kthvalue(
                rand_vals, num_features - num_selected_features
            ).values.item()
            yield (rand_vals > threshold).to(device=device).long()

    ★ 要点：
      1. ★ 前两个样本是**固定端点**：全 1（原始输入）与全 0（baseline），
         **无条件**产出，不消耗随机预算。
      2. ★ 阈值取的是 **第 (M - k) 小**的值（`kthvalue(rand_vals, M-k)`），
         不是第 k 大 —— 因为 `kthvalue` 的 k 是「从小到大数的第几个」。
      3. ★ 比较是**严格大于** `rand_vals > threshold`。
         若改成 `>=`，当恰好有 M-k 个值等于阈值时，会多选一个。
      4. 选中的是**严格大于阈值**的那些 ⇒ 通常正好 k 个；
         但若并列（多个值等于阈值），严格大于会**少选**。
      5. `rand_vals` 是外部给定的（官方是 `torch.randn`），
         本题把它作显式入参 ⇒ 全部确定性。

    返回：
      {"sample": [...], "threshold": float, "num_selected": int,
       "endpoint": bool}
    """
    M = _check_pos_int(num_interp_features, "num_interp_features")
    k = _check_pos_int(num_selected_features, "num_selected_features")

    vals = [_to_frac(v) for v in _flat(rand_vals)]
    if len(vals) != M:
        raise ValueError(
            "rand_vals length %d must equal num_interp_features %d"
            % (len(vals), M))

    # threshold = kthvalue(rand_vals, M - k)：从小到大数的第 (M-k) 个
    # ★ 官方 kthvalue 的 k 是 1-based ⇒ 第 (M-k) 小
    kth = M - k
    if kth < 1 or kth > M:
        raise ValueError(
            "kthvalue index %d out of range for %d values" % (kth, M))
    ordered = sorted(vals)
    threshold = ordered[kth - 1]

    # ★ 严格大于
    sample = [1 if v > threshold else 0 for v in vals]

    return {
        "sample": sample,
        "threshold": _f(threshold),
        "num_selected": sum(sample),
        "endpoint": False,
    }


def perturb_endpoints(num_interp_features):
    """官方 perturb generator 的两个固定端点样本。

    返回：
      {"samples": [[1]*M, [0]*M], "num_endpoints": 2}
    """
    M = _check_pos_int(num_interp_features, "num_interp_features")
    return {
        "samples": [[1] * M, [0] * M],
        "num_endpoints": 2,
    }


def perturb_sequence(rand_vals_list, selected_counts, num_interp_features):
    """官方 perturb generator 的完整序列（前两个端点 + 随机样本）。

    `rand_vals_list` 与 `selected_counts` 逐项对应；
    长度相同，第 i 项 rand_vals 用第 i 个 selected_count 生成。

    返回：
      {"samples": [[...], ...], "num_endpoints": 2, "num_random": int}
    """
    M = _check_pos_int(num_interp_features, "num_interp_features")
    if len(rand_vals_list) != len(selected_counts):
        raise ValueError(
            "rand_vals_list length %d must equal selected_counts length %d"
            % (len(rand_vals_list), len(selected_counts)))

    out = [[1] * M, [0] * M]
    for rv, k in zip(rand_vals_list, selected_counts):
        out.append(perturb_sample(rv, k, M)["sample"])
    return {
        "samples": out,
        "num_endpoints": 2,
        "num_random": len(rand_vals_list),
    }


# --------------------------------------------------------------------------
# mode 4: n_evaluations
# --------------------------------------------------------------------------


def n_evaluations(total_features, n_samples, perturbations_per_eval,
                  exact=False):
    """前向评估次数的两套口径。

    官方实现：

      # captum/attr/_core/shapley_value.py::ShapleyValueSampling
      def _get_n_evaluations(self, total_features, n_samples,
                              perturbations_per_eval):
          return math.ceil(total_features / perturbations_per_eval) * n_samples

      # captum/attr/_core/shapley_value.py::ShapleyValues
      def _get_n_evaluations(self, total_features, n_samples,
                              perturbations_per_eval):
          return math.ceil(total_features / perturbations_per_eval) * math.factorial(
              total_features
          )

    ★ 要点：
      1. 采样版是 `ceil(total_features / perturbations_per_eval) * n_samples`
         —— **先向上取整再乘**。
      2. ★ 精确版（`ShapleyValues`）把 `n_samples` **完全忽略**，
         换成 `factorial(total_features)` —— 因为它枚举了全部 M! 个排列。
      3. `perturbations_per_eval` 只出现在 `ceil` 的分母里，
         **不**影响乘法因子。
      4. `ShapleyValues` 的 `ceil(total/ppe)` 也保留了 —— 也就是
         「每个排列的前向次数」除以批大小向上取整。

    返回：
      {"n_evaluations": int, "mode": "exact" | "sampling",
       "permutation_factor": int}
    """
    total = _check_pos_int(total_features, "total_features")
    ns = _check_pos_int(n_samples, "n_samples")
    ppe = _check_pos_int(perturbations_per_eval, "perturbations_per_eval")

    batches = math.ceil(total / ppe)
    if exact:
        factor = math.factorial(total)
        mode = "exact"
    else:
        factor = ns
        mode = "sampling"
    return {
        "n_evaluations": batches * factor,
        "mode": mode,
        "permutation_factor": factor,
    }


def all_permutations(num_features, num_samples=None):
    """官方 _all_perm_generator 的枚举口径。

    官方实现（captum/attr/_core/shapley_value.py）：

        def _all_perm_generator(num_features, num_samples):
            for perm in itertools.permutations(range(num_features)):
                yield perm

    ★ 要点：
      1. ★ `num_samples` 参数被**完全忽略** —— 生成器产出全部 M! 个排列，
         与传进来的 num_samples 无关。
      2. 排列顺序是 `itertools.permutations` 的字典序。

    返回：
      {"permutations": [[...], ...], "count": int, "num_samples_ignored": bool}
    """
    M = _check_pos_int(num_features, "num_features")
    import itertools
    perms = [list(p) for p in itertools.permutations(range(M))]
    return {
        "permutations": perms,
        "count": len(perms),
        "num_samples_ignored": True,
    }


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
        if kw.get("kind") == "endpoints":
            return perturb_endpoints(kw["num_interp_features"])
        if kw.get("kind") == "sequence":
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
        return all_permutations(
            kw["num_features"], kw.get("num_samples"),
        )
    raise ValueError("unknown mode: %r" % (mode,))
