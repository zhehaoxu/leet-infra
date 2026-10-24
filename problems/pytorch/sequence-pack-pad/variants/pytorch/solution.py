import math

import torch


def solve(seqs: list, block_size: int, pad_token_id: int) -> dict:
    toks, pos, seg = [], [], []
    for d, s in enumerate(seqs):
        for j, t in enumerate(s):
            toks.append(t)
            pos.append(j)
            seg.append(d)
    total = len(toks)
    num_blocks = math.ceil(total / block_size)
    padn = num_blocks * block_size - total
    toks += [pad_token_id] * padn
    pos += [0] * padn
    seg += [-1] * padn
    am = [0 if v == -1 else 1 for v in seg]

    def reshape(x):
        return torch.tensor(x, dtype=torch.long).view(num_blocks, block_size)

    return {
        "input_ids": reshape(toks),
        "position_ids": reshape(pos),
        "segment_ids": reshape(seg),
        "attention_mask": reshape(am),
    }
