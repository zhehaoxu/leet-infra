"""参考答案：逐行按后继位掩码左移填充 labels。"""

import torch


def solve(input_ids, supervised_mask, ignore_index=-100):
    B, L = input_ids.shape
    labels = torch.full((B, L), ignore_index, dtype=torch.long)
    for b in range(B):
        for i in range(L - 1):
            if float(supervised_mask[b, i + 1]) > 0:
                labels[b, i] = int(input_ids[b, i + 1])
    return labels
