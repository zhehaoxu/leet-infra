import torch


def _sep_set(sep_ids):
    values = sep_ids.tolist() if torch.is_tensor(sep_ids) else list(sep_ids)
    out = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("sep_ids 必须只含 int")
        out.add(int(value))
    return out


def solve(token_ids, sep_ids, keys, values, init_cache_size=4, sep_cache_size=32,
          local_size=224, boundary=0):
    """SepLLM 单步压缩：sink + 分隔符区 + 本地窗（float64 内部，按输入 dtype 返回）。"""
    if token_ids.dim() != 1 or token_ids.dtype != torch.int64:
        raise ValueError("token_ids 必须是一维 int64")
    seq_len = int(token_ids.shape[0])
    if seq_len < 1 or len(keys) != len(values) or len(keys) < 1:
        raise ValueError("token_ids 不能为空，keys/values 必须逐层一一对应")
    out_dtype = keys[0].dtype
    heads, head_dim = int(keys[0].shape[0]), int(keys[0].shape[2])
    for k, v in zip(keys, values):
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError("每层 K/V 必须是三维 (H, L, D)")
        if k.shape[1] != seq_len or v.shape[1] != seq_len:
            raise ValueError("每层 K/V 的 seq 维必须等于 len(token_ids)")
        if (k.shape[0], k.shape[2]) != (heads, head_dim) or v.shape != k.shape:
            raise ValueError("各层 (H, D) 必须一致")
        if k.dtype != out_dtype or v.dtype != out_dtype or out_dtype not in (torch.float32, torch.float64):
            raise ValueError("K/V 必须同 dtype 且为 float32/float64")
    for name, value in (("init_cache_size", init_cache_size), ("sep_cache_size", sep_cache_size),
                        ("local_size", local_size), ("boundary", boundary)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是 int")
    if init_cache_size < 1 or sep_cache_size < 1 or local_size < 1:
        raise ValueError("init_cache_size/sep_cache_size/local_size 必须 >= 1")
    if boundary < 0 or 0 < boundary < init_cache_size:
        raise ValueError("boundary 必须为 0 或 >= init_cache_size")

    right = init_cache_size if boundary == 0 else boundary      # 首次压缩：分隔符区为空
    if seq_len - local_size <= right:
        raise ValueError("需要 seq_len - local_size > boundary（上游 assert）")

    ids = token_ids.tolist()
    sep_set = _sep_set(sep_ids)
    sep_all = list(range(init_cache_size, right))               # 累积区：上游直接切片
    sep_all += [p for p in range(right, seq_len - local_size) if ids[p] in sep_set]
    if len(sep_all) > sep_cache_size:
        sep_all = sep_all[-sep_cache_size:]                     # 只留最新的 s 个，边界锁死
    kept = list(range(init_cache_size)) + sep_all + list(range(seq_len - local_size, seq_len))
    index = torch.tensor(kept, dtype=torch.int64)
    return {
        "kept": index,
        "token_ids": token_ids.index_select(0, index),
        "keys": [k.to(torch.float64).index_select(1, index).to(out_dtype) for k in keys],
        "values": [v.to(torch.float64).index_select(1, index).to(out_dtype) for v in values],
        "boundary": init_cache_size + len(sep_all),
    }
