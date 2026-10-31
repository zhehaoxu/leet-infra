import torch
import torch.nn.functional as F


def _check_num(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} 必须是有限实数")
    return v


def _check_heads(name, heads):
    if not isinstance(heads, (list, tuple)) or not heads:
        raise ValueError(f"{name} 必须是非空 list")
    out, n, d = [], None, None
    for h, head in enumerate(heads):
        if not isinstance(head, (list, tuple)) or not head:
            raise ValueError(f"{name}[{h}] 必须是非空 list")
        if n is None:
            n = len(head)
        elif len(head) != n:
            raise ValueError(f"{name} 各头的 token 数必须一致")
        row = []
        for vec in head:
            if not isinstance(vec, (list, tuple)) or not vec:
                raise ValueError(f"{name}[{h}] 的向量必须是非空 list")
            if d is None:
                d = len(vec)
            elif len(vec) != d:
                raise ValueError(f"{name} 的向量维度必须一致")
            row.append([_check_num(f"{name}[{h}]", x) for x in vec])
        out.append(row)
    return out, n, d


def solve(key_states, query_states, num_key_value_groups, topk=1024,
          kernel_size=5, norm=1.0):
    """GemFilter 早层输入裁剪的打分与选择（2409.17422，CPU 参考口径）。"""
    kv_keys, n, d = _check_heads("key_states", key_states)
    q_heads, q_len, qd = _check_heads("query_states", query_states)
    kv_heads = len(kv_keys)
    num_heads = len(q_heads)
    if qd != d:
        raise ValueError("Q 与 K 的向量维度必须一致")
    if isinstance(num_key_value_groups, bool) or not isinstance(num_key_value_groups, int) \
            or num_key_value_groups < 1:
        raise ValueError("num_key_value_groups 必须是 >= 1 的 int")
    if kv_heads * num_key_value_groups != num_heads:
        raise ValueError("num_heads 必须等于 kv_heads * num_key_value_groups")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError("topk 必须是 >= 1 的 int")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) \
            or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size 必须是正奇数")
    norm = _check_num("norm", norm)
    if norm <= 0:
        raise ValueError("norm 必须为正")

    # repeat_kv：repeat_interleave 语义（每个 KV 头连续复制 groups 份）
    keys = torch.tensor(kv_keys, dtype=torch.float32).repeat_interleave(
        num_key_value_groups, dim=0)                       # [H, n, d]
    queries = torch.tensor(q_heads, dtype=torch.float32)
    query_last = queries[:, -1:, :]                        # 上游只取最后一个 query 位置

    # QK 内积 -> 取唯一的 query 行 -> 头间求和
    inner = torch.matmul(query_last, keys.transpose(-1, -2))[:, 0, :]      # [H, n]
    summed = inner.sum(dim=0)                              # [n]

    # avg_pool1d：两端零填充、分母恒为 kernel_size
    pooled_t = F.avg_pool1d(
        summed[None, None, :], kernel_size=kernel_size,
        padding=kernel_size // 2, stride=1)[0, 0]
    pooled = pooled_t.tolist()

    # top-k：值降序、同值按下标升序；k_eff = min(topk, n)
    k_eff = min(topk, n)
    order = sorted(range(n), key=lambda j: (-pooled[j], j))[:k_eff]
    distances = [pooled[j] if norm == 1.0 else pooled[j] / norm for j in order]

    return {"indices": order, "distances": distances, "pooled": pooled}
