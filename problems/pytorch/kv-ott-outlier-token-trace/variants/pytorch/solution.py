"""参考解：OTT 的离群 token 追踪（照 models/utils_OTT.py::quant_step L104-140 的记账）。

量化本体（triton_quantize_and_pack_along_last_dim）与 quant-kivi-kv-asym 重复，不进本题；
sink_k / sink_v 只服务 CUDA 侧注意力回算，与追踪决策无关，也剥掉。
"""

import torch


def solve(keys, values, quantized_len, sink_id, sink_scores, group_size=128,
          residual_length=128, sink_num=5, sink_max_size=32):
    for t in (keys, values):
        if not torch.is_tensor(t) or t.dim() != 4:
            raise ValueError("keys/values 必须是四维 (1, H, F, D)")
        if t.dtype not in (torch.float32, torch.float64):
            raise ValueError("keys/values 必须是 float32/float64")
    if keys.shape != values.shape:
        raise ValueError("keys 与 values 的形状必须一致")
    if keys.shape[0] != 1:
        raise ValueError("batch 维必须是 1（上游只写 keys[0]）")
    if keys.dtype != values.dtype:
        raise ValueError("keys 与 values 的 dtype 必须一致")
    if not torch.is_tensor(sink_id) or sink_id.dim() != 1:
        raise ValueError("sink_id 必须是一维张量")
    if not torch.is_tensor(sink_scores) or sink_scores.dim() != 1:
        raise ValueError("sink_scores 必须是一维张量")
    if sink_id.dtype != torch.int64:
        raise ValueError("sink_id 必须是 int64")
    if sink_scores.dtype != keys.dtype:
        raise ValueError("sink_scores 的 dtype 必须与 keys 一致")
    if sink_id.shape[0] != sink_scores.shape[0]:
        raise ValueError("sink_id 与 sink_scores 的长度必须一致")
    if sink_id.numel() and sink_id.numel() < sink_num:
        raise ValueError("已追踪条目不足 sink_num 个时上游取 sink_scores[sink_num-1] 会越界")
    for name, v in (("quantized_len", quantized_len), ("group_size", group_size),
                    ("residual_length", residual_length), ("sink_num", sink_num),
                    ("sink_max_size", sink_max_size)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} 必须是整数")
    if quantized_len < 0:
        raise ValueError("quantized_len 不能为负")
    if group_size < 1 or sink_num < 1 or sink_max_size < 1:
        raise ValueError("group_size / sink_num / sink_max_size 必须是正整数")
    if residual_length < 0:
        raise ValueError("residual_length 不能为负")

    n, h, f, d = keys.shape
    quant_len = max(0, (f - residual_length) // group_size * group_size)   # 上游 L104
    chunk_k = keys[:, :, :quant_len, :]
    chunk_v = values[:, :, :quant_len, :]
    if quant_len == 0:                                   # 上游 L361 的长度门槛没到
        return {"quant_len": torch.tensor(0, dtype=torch.int64),
                "scores": chunk_k[0].abs().mean(dim=(0, 2)),
                "new_indices": torch.zeros(0, dtype=torch.int64),
                "sink_id": sink_id.clone(), "sink_scores": sink_scores.clone(),
                "keys_out": chunk_k.clone(), "values_out": chunk_v.clone()}
    if sink_num > quant_len:
        raise ValueError("sink_num 不能超过本次可量化的 token 数")

    scores = chunk_k[0].abs().mean(dim=(0, 2))           # metric(k, v)，与 v 无关

    if sink_id.numel() == 0:                             # 首次：整段取最小的 sink_num 个
        order = torch.argsort(scores, dim=0, stable=True)[:sink_num]
        new_local, new_scores = order, scores[order]
    else:
        threshold = sink_scores[sink_num - 1]
        mask = torch.nonzero(scores < threshold, as_tuple=False).squeeze(1)
        if mask.numel() == 0 or sink_id.numel() >= sink_max_size:
            new_local = torch.zeros(0, dtype=torch.int64)
            new_scores = torch.zeros(0, dtype=scores.dtype)
        else:
            cand = scores[mask]
            order = torch.argsort(cand, dim=0, stable=True)[:sink_num]
            new_local, new_scores = mask[order], cand[order]

    if new_local.numel():
        kept_id = torch.cat([sink_id, new_local + quantized_len])
        kept_scores = torch.cat([sink_scores, new_scores])
        perm = torch.argsort(kept_scores, dim=0, stable=True)
        sink_id, sink_scores = kept_id[perm], kept_scores[perm]
        chunk_k = chunk_k.clone()
        chunk_v = chunk_v.clone()
        # 上游 L139-140：切片 dim=1 是本次选中的那批 token，均值跨 token、逐 head 广播回填
        # 广播的是同一个代表向量，所以取数用分数序（上游）还是下标序都同结果
        chunk_k[0, :, new_local, :] = chunk_k[0, :, new_local, :].mean(dim=1, keepdim=True)
        chunk_v[0, :, new_local, :] = chunk_v[0, :, new_local, :].mean(dim=1, keepdim=True)

    return {"quant_len": torch.tensor(quant_len, dtype=torch.int64),
            "scores": scores, "new_indices": new_local,
            "sink_id": sink_id, "sink_scores": sink_scores,
            "keys_out": chunk_k, "values_out": chunk_v}
