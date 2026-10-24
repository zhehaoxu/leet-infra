import torch


# 独立重放参考：逐张量重新实现同一调度（判题只认题面规定调度）
def _ref(init):
    W = init.shape[0]
    b = [[float(init[q][c]) for c in range(W)] for q in range(W)]
    out = []
    for t in range(W - 1):
        nxt = {}
        for q in range(W):
            s = (q - 1 - t) % W
            nxt[(q + 1) % W] = (s, b[q][s])
        for r, (s, val) in nxt.items():
            b[r][s] += val
        out.append([[b[q][c] for c in range(W)] for q in range(W)])
    for t in range(W - 1):
        nxt = {}
        for q in range(W):
            s = (q - t) % W
            nxt[(q + 1) % W] = (s, b[q][s])
        for r, (s, val) in nxt.items():
            b[r][s] = val
        out.append([[b[q][c] for c in range(W)] for q in range(W)])
    return out


def _full(init):
    W = init.shape[0]
    return [float(sum(init[q][c].item() for q in range(W))) for c in range(W)]


# 样例用例
def check_example(solve):
    init = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    snaps = solve(init)
    assert len(snaps) == 4
    expect = [
        [[1, 10, 3], [4, 5, 9], [11, 8, 9]],
        [[12, 10, 3], [4, 15, 9], [11, 8, 18]],
        [[12, 10, 18], [12, 15, 9], [11, 15, 18]],
        [[12, 15, 18], [12, 15, 18], [12, 15, 18]],
    ]
    for sn, ex in zip(snaps, expect):
        assert sn.shape == (3, 3)
        assert torch.equal(sn, torch.tensor(ex, dtype=sn.dtype)), sn.tolist()


def check_len_and_shape(solve):
    for W in (2, 3, 4, 5):
        torch.manual_seed(100 + W)
        init = torch.randint(-5, 6, (W, W)).float()
        snaps = solve(init)
        assert len(snaps) == 2 * (W - 1), (W, len(snaps))
        for sn in snaps:
            assert sn.shape == (W, W)


def check_final_is_allreduce(solve):
    # 末快照每行都应等于逐块求和 full
    for W in (2, 3, 4, 6):
        torch.manual_seed(W)
        init = torch.randint(-9, 10, (W, W)).float()
        snaps = solve(init)
        full = torch.tensor(_full(init))
        for row in snaps[-1]:
            assert torch.allclose(row, full, atol=1e-6), (W, row.tolist(), full.tolist())


def check_diagonal_after_rs(solve):
    # reduce-scatter 结束后（第 W-1 个快照）buf[q][q] == full[q]
    for W in (3, 4, 5):
        torch.manual_seed(7 * W)
        init = torch.randint(-4, 5, (W, W)).float()
        snaps = solve(init)
        rs_end = snaps[W - 2]
        full = _full(init)
        for q in range(W):
            assert abs(rs_end[q][q].item() - full[q]) < 1e-6, (W, q, rs_end.tolist(), full)


def check_matches_ref(solve):
    for W in (2, 3, 4, 5, 7):
        torch.manual_seed(31 * W + 1)
        init = torch.randint(-8, 9, (W, W)).float()
        snaps = solve(init)
        ref = _ref(init)
        assert len(snaps) == len(ref)
        for sn, rf in zip(snaps, ref):
            assert torch.allclose(sn, torch.tensor(rf), atol=1e-6), (W, sn.tolist(), rf)


def check_world_one_empty(solve):
    snaps = solve(torch.tensor([[3.] ]))
    assert isinstance(snaps, list) and len(snaps) == 0, snaps


def check_input_not_mutated(solve):
    init = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    snap = init.clone()
    solve(init)
    assert torch.equal(init, snap), "不应修改输入"


check_example.example = True
