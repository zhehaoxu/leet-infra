import torch


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _check_lens(lens, name, limit):
    """校验 lens 序列：非负 int 且 <= limit；返回原列表（None 原样返回）。"""
    if lens is None:
        return None
    if isinstance(lens, (int, torch.Tensor)) or not hasattr(lens, "__len__"):
        raise ValueError(f"{name} 必须是 None 或 int 序列，got {lens!r}")
    out = []
    for v in lens:
        if not _is_int(v) or v < 0:
            raise ValueError(f"{name} 各项必须是非负 int，got {v!r}")
        if v > limit:
            raise ValueError(f"{name} 各项不能超过 {name[:-5]}={limit}，got {v}")
        out.append(v)
    return out


def solve(q_len, kv_len, seq_lens=None, kv_lens=None, kv_offset=0,
          causal=True, window=None, dtype=torch.float32):
    """契约见 starter.py docstring：可看处 0.0，遮蔽处 finfo.min。"""
    if not _is_int(q_len) or q_len < 0:
        raise ValueError(f"q_len 必须是非负 int，got {q_len!r}")
    if not _is_int(kv_len) or kv_len < 0:
        raise ValueError(f"kv_len 必须是非负 int，got {kv_len!r}")
    if not _is_int(kv_offset) or kv_offset < 0:
        raise ValueError(f"kv_offset 必须是非负 int，got {kv_offset!r}")
    if window is not None and (not _is_int(window) or window <= 0):
        raise ValueError(f"window 必须是 None 或 >=1 的 int，got {window!r}")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype 仅支持 float32 / float64，got {dtype}")
    seq_lens = _check_lens(seq_lens, "seq_lens", q_len)
    kv_lens = _check_lens(kv_lens, "kv_lens", kv_len)

    b = max(len(seq_lens) if seq_lens is not None else 0,
            len(kv_lens) if kv_lens is not None else 0, 1)
    if seq_lens is not None and len(seq_lens) not in (1, b):
        raise ValueError(f"seq_lens 长度须为 1 或 B={b}，got {len(seq_lens)}")
    if kv_lens is not None and len(kv_lens) not in (1, b):
        raise ValueError(f"kv_lens 长度须为 1 或 B={b}，got {len(kv_lens)}")

    def expand(lens):
        if lens is None:
            return None
        if len(lens) == 1:
            lens = lens * b  # 沿 batch 广播
        return torch.tensor(lens, dtype=torch.long)

    visible = torch.ones(b, q_len, kv_len, dtype=torch.bool)
    idx_i = torch.arange(q_len).view(1, q_len, 1)
    idx_j = torch.arange(kv_len).view(1, 1, kv_len)
    if causal:
        visible = visible & (idx_j <= idx_i + kv_offset)      # (1,q_len,kv_len)
    if window is not None:
        visible = visible & ((idx_i - (idx_j - kv_offset)) < window)
    kvl = expand(kv_lens)
    if kvl is not None:
        visible = visible & (idx_j < kvl.view(b, 1, 1))       # (B,1,kv_len)
    sql = expand(seq_lens)
    if sql is not None:
        visible = visible & (idx_i < sql.view(b, 1, 1))       # (B,q_len,1)

    mask = torch.zeros(b, q_len, kv_len, dtype=dtype)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(1)
