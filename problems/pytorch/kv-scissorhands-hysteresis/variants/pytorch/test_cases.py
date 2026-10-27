import torch


def _oracle(scores, tau, patience):
    """逐标量重算：python 计数循环，与参考解的向量化 where 路径结构不同。"""
    s_steps, length = scores.shape
    tau = float(tau)
    streak = [0] * length
    evict = [-1] * length
    keep = [[True] * length for _ in range(s_steps)]
    for s in range(s_steps):
        for j in range(length):
            if evict[j] >= 0:
                keep[s][j] = False
                continue
            streak[j] = streak[j] + 1 if float(scores[s, j]) < tau else 0
            if streak[j] >= patience:
                evict[j] = s
                keep[s][j] = False
    return torch.tensor(evict, dtype=torch.long), torch.tensor(keep, dtype=torch.bool)


def check_example(solve):
    torch.manual_seed(0)
    scores = torch.tensor([[0.9, 0.1, 0.6, 0.2],
                           [0.8, 0.2, 0.4, 0.3],
                           [0.7, 0.3, 0.6, 0.9],
                           [0.6, 0.4, 0.5, 0.8]])
    evict, keep = solve(scores, 0.5, 2)
    assert evict.tolist() == [-1, 1, -1, 1], f"got {evict.tolist()}"
    assert keep[0].tolist() == [True] * 4
    assert keep[1].tolist() == [True, False, True, False], f"got {keep[1].tolist()}"
    assert keep[3].tolist() == [True, False, True, False]


check_example.example = True


def check_single_dip_survives(solve):
    """单帧低分不驱逐：patience=2 且每次低分后立即回升 → 全部存活。"""
    torch.manual_seed(1)
    scores = torch.tensor([[0.1, 0.9, 0.9],
                           [0.9, 0.1, 0.9],
                           [0.9, 0.9, 0.1],
                           [0.9, 0.9, 0.9]])
    evict, keep = solve(scores, 0.5, 2)
    assert evict.tolist() == [-1, -1, -1], f"单帧低分不应驱逐，got {evict.tolist()}"
    assert bool(keep.all())


def check_consecutive_run_triggers(solve):
    """必须连续 patience 步低分才驱逐：m=3 时 2 连低 + 回升不驱逐，3 连低驱逐。"""
    torch.manual_seed(2)
    scores = torch.tensor([[0.1, 0.1],
                           [0.1, 0.1],
                           [0.9, 0.1],
                           [0.9, 0.1],
                           [0.9, 0.1]])
    evict, _ = solve(scores, 0.5, 3)
    assert evict[0].item() == -1, "token0 只有 2 连低，不应驱逐"
    assert evict[1].item() == 2, f"token1 连续低分，应在第 3 步（index 2）驱逐，got {evict[1].item()}"


def check_threshold_is_strict(solve):
    """score == tau 不算低分；τ 边界不得用 <= 。"""
    torch.manual_seed(3)
    scores = torch.full((3, 2), 0.5)
    evict, keep = solve(scores, 0.5, 2)
    assert evict.tolist() == [-1, -1], f"got {evict.tolist()}"
    assert bool(keep.all())


def check_matches_oracle_and_errors(solve):
    """随机用例与逐标量计数对拍，并校验非法 patience。"""
    torch.manual_seed(21)
    scores = torch.rand(6, 9)
    for tau, m in ((0.3, 2), (0.6, 4)):
        exp_e, exp_k = _oracle(scores, tau, m)
        e, k = solve(scores, tau, m)
        assert torch.equal(e, exp_e), f"tau={tau},m={m} 驱逐步与重算不符"
        assert torch.equal(k, exp_k), f"tau={tau},m={m} 掩码与重算不符"
    try:
        solve(scores, 0.5, 0)
        raise AssertionError("patience=0 应抛 ValueError")
    except ValueError:
        pass
