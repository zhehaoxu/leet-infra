"""参考答案：Qwen2-VL 风格 M-RoPE 位置编号（文本/图像块顺序扫描）。"""

import torch


def solve(mm_token_type, grid_thw):
    L = mm_token_type.shape[0]
    pos = torch.zeros(3, L, dtype=torch.long)
    running = 0
    img_ptr = 0
    i = 0
    while i < L:
        if int(mm_token_type[i].item()) == 0:
            pos[:, i] = running
            running += 1
            i += 1
        else:
            h = int(grid_thw[img_ptr][1].item())
            w = int(grid_thw[img_ptr][2].item())
            for k in range(h * w):
                r, c = divmod(k, w)
                pos[0, i + k] = running
                pos[1, i + k] = running + r
                pos[2, i + k] = running + c
            running += max(h, w)
            i += h * w
            img_ptr += 1
    return pos
