import torch


def solve(descriptors, scores, bins, ranges):
    # TODO: MAP-Elites 质量-多样性档案的插入账。
    #   - descriptors (N, D) 是行为描述子，scores (N,) 是该候选的适应度。
    #   - 每维按 [lo, hi] 均分成 bins[k] 格：slot = floor((v - lo) / (hi - lo) * bins[k])，
    #     恰好等于 hi 时夹到 bins[k] - 1（不丢）。
    #   - 描述子越界（v < lo 或 v > hi）的候选**直接丢弃**，cells 记 -1。
    #   - flat 下标行优先：flat = ((i0*b1) + i1)*b2 + i2 ...
    #   - 只有 score **严格大于**该格当前精英才替换；同分保留先到者。
    #   返回 {"cells", "accepted", "elite_score", "elite_index", "occupancy",
    #         "best_score", "best_index"}；空格精英分 -inf、下标 -1。
    raise NotImplementedError
