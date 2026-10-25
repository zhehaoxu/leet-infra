"""参考答案：多 episode 轨迹拼装（分段 → 训练序重排 → done/bootstrap 三分规则）。

向量化路径：cumsum 标段号、bincount 数长度、argsort 定段序、一次 gather 完成重排；
next_values 先按出现序做「roll + 段末覆盖」，再统一收拢到训练序。
"""

import torch


def solve(episode_id, terminated, truncated, actions, rewards, values, final_value):
    T = int(episode_id.numel())
    if T == 0:
        e64 = torch.zeros(0, dtype=torch.long)
        e32 = torch.zeros(0, dtype=torch.float32)
        return (e64.clone(), e32.clone(), e32.clone(), e64.clone(),
                e32.clone(), e64.clone(), e64.clone())

    ep = episode_id.to(torch.long)
    # 1) 分段：episode_id 的极大连续段；run_idx[t] = 第 t 步所属「出现段」序号 0..K-1
    new_run = torch.ones(T, dtype=torch.bool)
    new_run[1:] = ep[1:] != ep[:-1]
    run_idx = torch.cumsum(new_run.to(torch.long), 0) - 1
    K = int(run_idx[-1].item()) + 1

    # 2) done：段末步（下一段起点或数组尾）
    is_last = torch.zeros(T, dtype=torch.bool)
    is_last[:-1] = new_run[1:]
    is_last[-1] = True
    dones = is_last.to(torch.long)

    # 3) 训练序 = 段按 episode_id 升序；段内保持时间序
    seg_ids = ep[new_run]                                  # (K,) 按出现顺序的段 id
    ep_order = torch.argsort(seg_ids)                      # 训练槽位 e -> 出现段
    run_to_slot = torch.empty(K, dtype=torch.long)
    run_to_slot[ep_order] = torch.arange(K, dtype=torch.long)

    lengths_by_run = torch.bincount(run_idx, minlength=K)  # (K,) 按出现段
    starts_by_run = torch.cumsum(lengths_by_run, 0) - lengths_by_run
    lengths = lengths_by_run[ep_order]
    starts = torch.cumsum(lengths, 0) - lengths

    # 4) 出现序 -> 训练序的收拢下标：gather[t] = 训练位 t 对应的出现下标
    idx = torch.arange(T, dtype=torch.long)
    train_pos = starts[run_to_slot[run_idx]] + (idx - starts_by_run[run_idx])
    gather = torch.empty(T, dtype=torch.long)
    gather[train_pos] = idx

    # 5) next_value 三分规则（先按出现序：roll 供段内后继，段末覆盖 bootstrap/0）
    vals = values.to(torch.float32)
    next_values = torch.zeros(T, dtype=torch.float32)
    next_values[:-1] = vals[1:]                            # 跨段泄漏会被下一步覆盖掉
    boot = final_value.to(torch.float32)[run_idx]          # 各步所在出现段的 V(s_end)
    last_trunc = truncated.to(torch.bool)[is_last]
    last_boot = boot[is_last]
    next_values[is_last] = torch.where(last_trunc, last_boot, torch.zeros_like(last_boot))

    return (actions.to(torch.long)[gather],
            rewards.to(torch.float32)[gather],
            vals[gather],
            dones[gather],
            next_values[gather],
            lengths,
            starts)
