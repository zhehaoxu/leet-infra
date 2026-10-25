"""参考答案：奖励归一化管道（账本 + 统计源选择 + 白化开关），全程 float64 累加、输出 float32。"""

import torch


def solve(reward_batches, group_ids, stat_source, momentum, whiten_std, eps):
    x = reward_batches.to(torch.float64)
    t_n, b_n = int(x.shape[0]), int(x.shape[1])
    g = group_ids.reshape(-1).to(torch.long)
    m = float(momentum)
    e = float(eps)
    whiten = int(whiten_std) == 1
    src = int(stat_source)

    norm = torch.zeros(t_n, b_n, dtype=torch.float64)
    out_rms = torch.zeros(t_n, dtype=torch.float64)
    rmean = torch.zeros(t_n + 1, dtype=torch.float64)
    rvar = torch.ones(t_n + 1, dtype=torch.float64)

    for t in range(t_n):
        row = x[t]
        b_mu = torch.mean(row)
        b_var = torch.mean((row - b_mu) ** 2)          # 总体方差 ddof=0
        if src == 1:                                   # 用「本批更新之前」的账本
            mu, var = rmean[t], rvar[t]
        elif src == 2:                                 # 批内组统计（GRPO 口径的组来源）
            n_groups = int(g.max()) + 1 if g.numel() else 0
            cnt = torch.zeros(n_groups, dtype=torch.float64).index_add_(
                0, g, torch.ones(b_n, dtype=torch.float64))
            gsum = torch.zeros(n_groups, dtype=torch.float64).index_add_(0, g, row)
            gmu = gsum / cnt
            gvar = torch.zeros(n_groups, dtype=torch.float64).index_add_(
                0, g, (row - gmu[g]) ** 2) / cnt
            mu, var = gmu[g], gvar[g]
        else:                                          # 批全局
            mu, var = b_mu, b_var

        centered = row - mu
        norm[t] = centered / torch.sqrt(var + e) if whiten else centered
        out_rms[t] = torch.sqrt(torch.mean(norm[t] ** 2))

        rmean[t + 1] = (1.0 - m) * rmean[t] + m * b_mu
        rvar[t + 1] = (1.0 - m) * rvar[t] + m * b_var

    return (norm.to(torch.float32), out_rms.to(torch.float32),
            rmean.to(torch.float32), rvar.to(torch.float32))
