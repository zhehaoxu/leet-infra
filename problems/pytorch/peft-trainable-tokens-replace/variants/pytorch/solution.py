import torch


def solve(weight, adapters):
    """TrainableTokens（huggingface/peft tuners/trainable_tokens）的 CPU 参考口径。

    weight [vocab, dim]（Embedding 权重）。adapters 为非空 list，每项是
    {"name": str, "token_indices": [int, ...], "delta": [[...], ...]}：
    delta 的第 k 行将**替换**（不是加到）weight 的第 token_indices[k] 行。
    任何两个 adapter 的 token_indices 相交 → ValueError（上游不支持重叠合并）。
    返回 merged_weight（依序 index_copy 后的权重）与
    originals（每个 adapter 命中的原始行，从最初 weight 抓取，供 unmerge 恢复）。
    """
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise ValueError("weight 必须是二维 tensor")
    if not torch.isfinite(weight).all():
        raise ValueError("weight 含非有限值")
    vocab, dim = weight.shape

    if not isinstance(adapters, (list, tuple)) or not adapters:
        raise ValueError("adapters 必须是非空 list")

    names, idx_lists, delta_lists = [], [], []
    for i, ad in enumerate(adapters):
        if not isinstance(ad, dict) or set(ad) != {"name", "token_indices", "delta"}:
            raise ValueError(f"adapters[{i}] 必须是含 name/token_indices/delta 的 dict")
        name = ad["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"adapters[{i}].name 必须是非空字符串")
        if name in names:
            raise ValueError(f"adapter 名重复：{name}")
        names.append(name)

        indices = ad["token_indices"]
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError(f"adapters[{i}].token_indices 必须是非空 list")
        seen = set()
        for j, t in enumerate(indices):
            if isinstance(t, bool) or not isinstance(t, int):
                raise ValueError(f"adapters[{i}].token_indices[{j}] 必须是 int")
            if not (0 <= t < vocab):
                raise ValueError(f"adapters[{i}].token_indices[{j}] 越界")
            if t in seen:
                raise ValueError(f"adapters[{i}].token_indices 内重复：{t}")
            seen.add(t)
        idx_lists.append(list(indices))

        delta = ad["delta"]
        if isinstance(delta, torch.Tensor):
            if delta.dim() != 2 or delta.shape != (len(indices), dim):
                raise ValueError(f"adapters[{i}].delta 形状须为 [{len(indices)}, {dim}]")
            if not torch.isfinite(delta).all():
                raise ValueError(f"adapters[{i}].delta 含非有限值")
            delta_lists.append(delta.clone())
        else:
            if not isinstance(delta, (list, tuple)) or len(delta) != len(indices):
                raise ValueError(f"adapters[{i}].delta 行数须等于 token_indices 长度")
            rows = []
            for j, row in enumerate(delta):
                if not isinstance(row, (list, tuple)) or len(row) != dim:
                    raise ValueError(f"adapters[{i}].delta[{j}] 长度须为 {dim}")
                for v in row:
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise ValueError(f"adapters[{i}].delta[{j}] 含非数值")
                    v = float(v)
                    if v != v or v in (float("inf"), float("-inf")):
                        raise ValueError(f"adapters[{i}].delta[{j}] 含非有限值")
                rows.append([float(v) for v in row])
            delta_lists.append(torch.tensor(rows, dtype=torch.float64).to(weight.dtype))

    # 不相交检查（上游 _check_overlapping_tokens：重叠合并未定义）
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if set(idx_lists[a]) & set(idx_lists[b]):
                raise ValueError(f"adapter {names[b]} 的 token indices 与 {names[a]} 重叠")

    # originals：update_layer 时从基座权重抓取的原始行（merge 前的 weight）
    originals = {
        name: weight[idx].clone() for name, idx in zip(names, idx_lists)
    }

    # 依序 index_copy：替换语义
    merged = weight.clone()
    for idx, delta in zip(idx_lists, delta_lists):
        index = torch.tensor(idx, dtype=torch.long)
        merged = merged.index_copy(dim=0, index=index, source=delta.to(merged.dtype))

    return {"merged_weight": merged, "originals": originals}
