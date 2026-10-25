"""参考答案：逐行找 assistant 极大段，按需丢弃段尾 EOS，段内置 1。"""

import torch


def solve(token_ids, role_ids, assistant_id, eos_id, supervise_eos=True):
    B, L = role_ids.shape
    mask = torch.zeros(B, L, dtype=torch.float32)
    for b in range(B):
        i = 0
        while i < L:
            if int(role_ids[b, i]) == assistant_id:
                j = i
                while j < L and int(role_ids[b, j]) == assistant_id:
                    j += 1
                end = j
                if (not supervise_eos) and end > i and int(token_ids[b, end - 1]) == eos_id:
                    end -= 1
                for k in range(i, end):
                    mask[b, k] = 1.0
                i = j
            else:
                i += 1
    return mask
