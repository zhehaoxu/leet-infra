"""参考答案：按定义的边界 b_r=floor(total*r/world) 算分片长度；逐参数区间判断是否有严格内部边界。"""

import torch


def solve(param_numels, world):
    total = int(sum(int(x) for x in param_numels))
    # 分片边界
    b = [(total * r) // world for r in range(world + 1)]
    per_rank = [b[r + 1] - b[r] for r in range(world)]

    # 逐参数区间 [s, e)，检查是否存在内部边界 b_r (1<=r<=world-1) 严格落在 (s, e)
    straddle = []
    s = 0
    for i, n in enumerate(param_numels):
        n = int(n)
        e = s + n
        for r in range(1, world):
            if s < b[r] < e:
                straddle.append(i)
                break
        s = e

    return (
        torch.tensor(per_rank, dtype=torch.long),
        torch.tensor(straddle, dtype=torch.long),
    )
