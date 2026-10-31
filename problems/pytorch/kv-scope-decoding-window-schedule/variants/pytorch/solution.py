"""参考解：SCOPE 解码期的窗口预算与不连续驱逐（逐行照抄上游 model/kv_utils.py）。

上游 SnapKVCluster.update_kv_in_decoding 的四个分支：'None' L518、'slide' L520-550、
'adaptive' L551-581、'discontinuous' L583-623，非法分支兜底 ValueError 在 L624；
类级计数器 current_decoding_step / jump_step / jump_layer 声明在 L438-440，
delta 由 run_longgenbench.py L195 从输出长度反推。两个计数器都在**每层调用**里 +1，
本题折叠成每 token 推进 num_hidden_layers 格；已用真实上游类（L 个实例 + 每层独立缓存）
在 13 组参数 x 4 个分支上逐槽位对拍一致。预填充段的 pooling/topk/聚类与
kv-snapkv-pooling 逐行重复，不进本题。
"""

import torch

METRICS = ("None", "slide", "adaptive", "discontinuous")
INT_NAMES = ("num_hidden_layers", "delta", "max_capacity_prompt",
             "decoding_recent_size", "decoding_window_size")
INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def solve(init_positions, attn, decoding_metric="discontinuous", num_hidden_layers=32,
          delta=15, max_capacity_prompt=512, decoding_recent_size=128,
          decoding_window_size=1024):
    if decoding_metric not in METRICS:
        raise ValueError("Decoding metric not supported")
    for name, v in zip(INT_NAMES, (num_hidden_layers, delta, max_capacity_prompt,
                                   decoding_recent_size, decoding_window_size)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} 必须是整数")
        if v < 1:
            raise ValueError(f"{name} 必须是正整数")
    if decoding_metric == "slide" and decoding_window_size <= decoding_recent_size:
        raise ValueError("slide 需要 decoding_window_size > decoding_recent_size")
    if not torch.is_tensor(attn) or attn.dim() != 3 or not attn.is_floating_point():
        raise ValueError("attn 必须是三维浮点张量")
    if not torch.is_tensor(init_positions) or init_positions.dim() != 1:
        raise ValueError("init_positions 必须是一维张量")
    if init_positions.dtype not in INT_DTYPES or init_positions.numel() == 0:
        raise ValueError("init_positions 必须是非空整数张量")
    steps, heads, width = attn.shape
    n_init = init_positions.numel()
    if steps < 1 or width != n_init + steps:
        raise ValueError("attn 的步数必须 >= 1 且宽等于 len(init_positions) + 步数")
    init = init_positions.tolist()
    if len(set(init)) != n_init or not all(0 <= p < width for p in init):
        raise ValueError("init_positions 必须互不相同且落在 [0, P + T) 内")
    if max_capacity_prompt > n_init:
        raise ValueError("max_capacity_prompt 不能超过初始槽位数")

    slots = [list(init) for _ in range(heads)]
    cds = 0
    jump = 0
    wins, lens, evict = [], [], []
    for t in range(steps):
        for h in range(heads):
            slots[h].append(n_init + t)
        n = len(slots[0])
        dw = 0
        evicted = False
        if decoding_metric == "slide":
            dw = decoding_window_size
            evicted = n >= max_capacity_prompt + dw
        elif decoding_metric in ("adaptive", "discontinuous"):
            dw = decoding_recent_size + cds // (delta * num_hidden_layers)
            cds += num_hidden_layers
            if n >= max_capacity_prompt + dw:
                if decoding_metric == "adaptive":
                    evicted = True
                elif jump >= delta * num_hidden_layers:
                    jump = 0
                    evicted = True
                else:
                    jump += num_hidden_layers
        wins.append(dw)
        if evicted:
            evict.append(t)
            keep = dw - decoding_recent_size
            tail = n - decoding_recent_size
            for h in range(heads):
                s = slots[h]
                region = s[max_capacity_prompt:tail]
                row = attn[t, h]
                order = sorted(range(len(region)), key=lambda i: (-float(row[region[i]]), i))
                slots[h] = s[:max_capacity_prompt] + [region[i] for i in order[:keep]] + s[tail:]
        lens.append(len(slots[0]))
    return {
        "window_sizes": torch.tensor(wins, dtype=torch.int64),
        "cache_lens": torch.tensor(lens, dtype=torch.int64),
        "evict_steps": torch.tensor(evict, dtype=torch.int64),
        "kept_positions": torch.tensor(slots, dtype=torch.int64),
    }
