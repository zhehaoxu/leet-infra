"""参考解：LogQuant 全精度索引调度（逐行照抄上游 src/LogQuant.py::LogQuantizedCache.update）。

上游在 layer_idx == 0 的那一次调用里维护两个「位置列表」：
  full_precision_logsparse_index  ←→ 本题 log_sparse_index
  local_slide_window_index        ←→ 本题 local_index
预填充段（一次进 L>1 个 token）走 L961-973，解码段（每次进 1 个 token）走 L976-982。
量化本体（quanto qint2/qint4 + q_group_size）与 quant-kivi-kv-asym 重复，不进本题。
"""

import torch


def solve(prefill_len, decode_steps, window_length):
    for name, v in (("prefill_len", prefill_len), ("decode_steps", decode_steps),
                    ("window_length", window_length)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} 必须是整数")
    if window_length < 1:
        raise ValueError("window_length 必须是正整数（为 0 时上游的折叠循环不会收敛）")
    if prefill_len < 0 or decode_steps < 0:
        raise ValueError("prefill_len 与 decode_steps 必须非负")
    if prefill_len == 0 and decode_steps == 0:
        raise ValueError("prefill_len 与 decode_steps 不能同时为 0")

    log, local = [], []
    seen = 0

    def on_step():                                  # 上游 L976-982 的单步解码规则
        nonlocal log, local, seen
        seen += 1
        local.append(seen - 1)
        if len(local) > 2 * window_length:
            log = (log + local[:window_length])[0::2] if log else local[:window_length]
            local = local[window_length:]

    if prefill_len > 1:
        seen += prefill_len
        local = list(range(prefill_len))            # 上游 L962-964
        tmp = log + local
        while len(tmp) >= 3 * window_length:        # 上游 L965-968：整段 2W 取偶数位
            log = tmp[0:2 * window_length:2]
            local = tmp[2 * window_length:]
            tmp = log + local
        if window_length < prefill_len < 3 * window_length:   # 上游 L969-971 的后置覆盖
            log, local = tmp[:window_length], tmp[window_length:]
        prefill_log = list(log)
        prefill_local = list(local)
    elif prefill_len == 1:                           # L==1 落进上游的 else 分支（按解码步处理）
        on_step()
        prefill_log, prefill_local = list(log), list(local)
    else:                                            # L==0：没有预填充调用，位置从第 0 个解码 token 起
        prefill_log, prefill_local = [], []

    for _ in range(decode_steps):
        on_step()

    fp = sorted(set(log) | set(local))
    return {
        "prefill_log_index": torch.tensor(prefill_log, dtype=torch.int64),
        "prefill_local_index": torch.tensor(prefill_local, dtype=torch.int64),
        "final_log_index": torch.tensor(log, dtype=torch.int64),
        "final_local_index": torch.tensor(local, dtype=torch.int64),
        "final_fp_positions": torch.tensor(fp, dtype=torch.int64),
    }
