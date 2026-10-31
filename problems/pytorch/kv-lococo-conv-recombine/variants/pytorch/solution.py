import torch
import torch.nn.functional as F


def solve(keys, values, scores, conv_weight, normalizer, hh_keep_rate, local_len, mem_size):
    """LoCoCo：H2O 重击 token + 卷积重算残差 + normalizer 混合（float64 内部）。"""
    if (keys.dim() != 3 or values.dim() != 3 or scores.dim() != 2
            or keys.shape != values.shape or scores.shape != (keys.shape[1], keys.shape[0])):
        raise ValueError("形状必须是 keys/values (S,H,D)、scores (H,S)")
    seq_len, num_heads, head_dim = keys.shape
    if keys.dtype not in (torch.float32, torch.float64) or not (
            keys.dtype == values.dtype == scores.dtype == conv_weight.dtype):
        raise ValueError("浮点入参必须同 dtype 且为 float32/float64")
    if type(mem_size) is not int or type(local_len) is not int:
        raise ValueError("mem_size/local_len 必须是 int")
    if type(hh_keep_rate) not in (int, float) or not 0.0 < float(hh_keep_rate) < 1.0:
        raise ValueError("hh_keep_rate 必须是 (0,1) 内的标量")
    rate = float(hh_keep_rate)
    mem_compress = int(mem_size * (1.0 - rate))        # 与上游一致：向零截断
    keep_hh = mem_size - mem_compress
    if (local_len < 0 or mem_size < 2 or mem_compress < 1
            or keep_hh < local_len or seq_len < keep_hh
            or conv_weight.dim() != 3
            or conv_weight.shape != (mem_compress, 2 * head_dim, conv_weight.shape[-1])):
        raise ValueError("mem_size/hh_keep_rate/local_len 或 conv_weight 形状非法")
    kernel = int(conv_weight.shape[-1])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("卷积核长度必须为正奇数")
    try:
        norm = float(normalizer)                       # 标量 float 或 0 维 tensor
    except (TypeError, ValueError):
        raise ValueError("normalizer 必须是标量")
    non_hh = seq_len - keep_hh
    if non_hh < mem_compress:
        raise ValueError("非重击 token 少于 mem_compress，无法选出残差")

    s64 = scores.to(torch.float64)
    cand = s64[:, : seq_len - local_len] if local_len else s64
    picked = torch.argsort(-cand, dim=-1, stable=True)[:, : keep_hh - local_len]  # 同分取小号
    hh_mask = torch.zeros_like(s64, dtype=torch.bool).scatter_(1, picked, True)
    if local_len:
        hh_mask[:, seq_len - local_len:] = True

    hh_idx = hh_mask.nonzero()[:, 1].reshape(num_heads, keep_hh)
    tok = torch.arange(seq_len).expand(num_heads, seq_len)
    nh_idx = tok.masked_select(~hh_mask).reshape(num_heads, non_hh)
    nh_scores = s64.masked_select(~hh_mask).reshape(num_heads, non_hh)
    top = torch.argsort(-nh_scores, dim=-1, stable=True)[:, :mem_compress]         # 同分取小号
    pos = top.sort(dim=-1).values                                                  # one-hot 行序
    res_idx = nh_idx.gather(1, pos)

    k64 = keys.to(torch.float64).permute(1, 2, 0)      # (H, D, S)
    v64 = values.to(torch.float64).permute(1, 2, 0)
    k_nh, v_nh = (x.masked_select((~hh_mask).unsqueeze(1).expand_as(x))
                  .reshape(num_heads, head_dim, non_hh) for x in (k64, v64))
    conv_prob = torch.softmax(F.relu(F.conv1d(torch.cat([k_nh, v_nh], dim=1),
                                              conv_weight.to(torch.float64),
                                              padding=(kernel - 1) // 2)), dim=-1)
    residual = F.one_hot(pos, non_hh).to(torch.float64)     # 每个残差行选中一个原 token
    blend = residual * (1.0 - norm) + conv_prob * norm

    hh_k, hh_v = (x.masked_select(hh_mask.unsqueeze(1).expand_as(x))
                  .reshape(num_heads, head_dim, keep_hh).transpose(1, 2) for x in (k64, v64))
    merged_keys = torch.cat([hh_k, torch.matmul(blend, k_nh.transpose(1, 2))], dim=1)
    merged_values = torch.cat([hh_v, torch.matmul(blend, v_nh.transpose(1, 2))], dim=1)
    out = keys.dtype
    return {"hh_idx": hh_idx, "res_idx": res_idx, "conv_prob": conv_prob.to(out),
            "blend": blend.to(out), "merged_keys": merged_keys.to(out),
            "merged_values": merged_values.to(out)}
