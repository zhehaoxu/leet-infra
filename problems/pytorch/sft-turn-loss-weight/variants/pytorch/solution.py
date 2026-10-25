"""参考答案：逐行扫描 assistant 极大段，段内 base/count 填充。"""

import torch


def solve(role_ids, assistant_id, turn_weights=None):
    B, L = role_ids.shape
    w = torch.zeros(B, L, dtype=torch.float32)
    for b in range(B):
        runs = []
        i = 0
        while i < L:
            if int(role_ids[b, i]) == assistant_id:
                j = i
                while j < L and int(role_ids[b, j]) == assistant_id:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        for t, (a, bb) in enumerate(runs):
            base = 1.0 if turn_weights is None else (
                float(turn_weights[t]) if t < len(turn_weights) else 1.0)
            n = bb - a
            for k in range(a, bb):
                w[b, k] = base / n
    return w
