import torch


def solve(weight, indices, values, scaling, x):
    """SHiRA（huggingface/peft tuners/shira）稀疏增量的 CPU 参考口径。

    weight [out, in]；indices 为 K 个 [row, col] 坐标（list 的二元组/list）；
    values 为 K 个数；scaling 非负实数，乘在每个 value 上。
    delta = 稀疏散点重建 [out, in]，重复坐标**累加**（torch.sparse_coo_tensor
    的 to_dense 语义，不是后者覆盖）；merged = weight + delta；y = x @ merged.T。
    """
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise ValueError("weight 必须是二维 tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("weight 含非有限值")
    out_features, in_features = weight.shape

    if not isinstance(indices, (list, tuple)) or len(indices) == 0:
        raise ValueError("indices 必须是非空 list")
    K = len(indices)
    # SHiRA 参数预算：K 个可训练稀疏权重不得多于基权重元素数
    if K > in_features * out_features:
        raise ValueError("稀疏参数量超过基权重元素数")
    rows, cols = [], []
    for i, pair in enumerate(indices):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"indices[{i}] 必须是 [row, col] 二元组")
        r, c = pair
        if isinstance(r, bool) or not isinstance(r, int):
            raise ValueError(f"indices[{i}] 行必须是 int")
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError(f"indices[{i}] 列必须是 int")
        if not (0 <= r < out_features):
            raise ValueError(f"indices[{i}] 行越界")
        if not (0 <= c < in_features):
            raise ValueError(f"indices[{i}] 列越界")
        rows.append(r)
        cols.append(c)

    if isinstance(values, torch.Tensor):
        if values.dim() != 1 or values.shape[0] != K:
            raise ValueError("values 必须是长度 K 的一维 tensor")
        vals = values.tolist()
    else:
        if not isinstance(values, (list, tuple)) or len(values) != K:
            raise ValueError("values 必须是长度 K 的 list")
        vals = list(values)
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"values[{i}] 必须是数")
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"values[{i}] 非有限")

    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        raise ValueError("scaling 必须是数")
    scaling = float(scaling)
    if scaling != scaling or scaling < 0 or scaling in (float("inf"),):
        raise ValueError("scaling 必须是非负有限数")

    if not isinstance(x, torch.Tensor) or x.dim() < 1:
        raise ValueError("x 必须是至少一维的 tensor")
    if x.shape[-1] != in_features:
        raise ValueError("x 末维须等于 in_features")
    if not torch.isfinite(x).all():
        raise ValueError("x 含非有限值")

    # 稀疏散点：重复坐标累加（等价 torch.sparse_coo_tensor(...).to_dense()）
    delta = torch.zeros(out_features, in_features, dtype=torch.float64)
    for (r, c), v in zip(zip(rows, cols), vals):
        delta[r, c] += float(v) * scaling
    delta = delta.to(weight.dtype)

    merged = weight + delta
    y = x @ merged.T
    return {
        "delta_weight": delta,
        "merged_weight": merged,
        "y": y,
    }
