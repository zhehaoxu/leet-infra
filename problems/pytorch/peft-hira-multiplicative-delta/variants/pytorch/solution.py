import torch


def _mat(t, name):
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"{name} 必须是二维 tensor")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} 含非有限值")
    return t


def solve(weight, hira_A, hira_B, x, fan_in_fan_out=False):
    """HiRA（huggingface/peft tuners/hira）乘性高秩适配的 CPU 参考口径。

    数学权重 W = weight（fan_in_fan_out=False，[out, in]）
                 或 weight.T（fan_in_fan_out=True，存储为 [in, out]）。
    delta = B @ A；有效权重 = W ⊙ delta；合并 = W + W ⊙ delta（乘性语义）。
    返回 dict 四个键，全部按输入存储布局返回；hira_result = x @ eff.T。
    """
    if not isinstance(fan_in_fan_out, bool):
        raise ValueError("fan_in_fan_out 必须是 bool")
    weight = _mat(weight, "weight")
    A = _mat(hira_A, "hira_A")
    B = _mat(hira_B, "hira_B")
    if not isinstance(x, torch.Tensor) or x.dim() < 1:
        raise ValueError("x 必须是至少一维的 tensor")
    if not torch.isfinite(x).all():
        raise ValueError("x 含非有限值")

    out_features, in_features = B.shape[0], A.shape[1]
    r = A.shape[0]
    if r < 1:
        raise ValueError("r 必须 >= 1")
    if A.shape != (r, in_features):
        raise ValueError("hira_A 形状须为 [r, in_features]")
    if B.shape != (out_features, r):
        raise ValueError("hira_B 形状须为 [out_features, r]")
    if x.shape[-1] != in_features:
        raise ValueError("x 末维须等于 in_features")

    if fan_in_fan_out:
        if weight.shape != (in_features, out_features):
            raise ValueError("fan_in_fan_out=True 时 weight 须为 [in, out]")
        w_math = weight.T                      # [out, in]
    else:
        if weight.shape != (out_features, in_features):
            raise ValueError("fan_in_fan_out=False 时 weight 须为 [out, in]")
        w_math = weight

    delta = B @ A                              # [out, in]
    eff = w_math * delta                       # 乘性：W ⊙ (B@A)
    hira_result = x @ eff.T
    merged = w_math + eff                      # W + W⊙Δ = W(1+Δ)

    if fan_in_fan_out:
        delta = delta.T
        eff = eff.T
        merged = merged.T
    return {
        "delta_weight": delta,
        "effective_weight": eff,
        "hira_result": hira_result,
        "merged_weight": merged,
    }
