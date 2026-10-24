import torch


def _oracle(hidden, ti, tw, E, EPD):
    """独立：显式按 (src,dst) 计数 + python 循环求 combine。"""
    T, H = hidden.shape
    k = ti.shape[1]
    P = E // EPD
    tpd = (T + P - 1) // P
    send = torch.zeros(P, P, dtype=torch.long)
    for t in range(T):
        src = min(t // tpd, P - 1)
        for j in range(k):
            dst = int(ti[t, j]) // EPD
            send[src, dst] += 1
    comb = hidden.float().clone()
    for t in range(T):
        acc = sum(float(tw[t, j]) * int(ti[t, j]) for j in range(k))
        comb[t] = comb[t] + acc
    return send, comb


def check_example(solve):
    hidden = torch.tensor([[1.0, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]])
    ti = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0],
                       [0, 2], [1, 3], [2, 0], [3, 1]])
    tw = torch.tensor([[.6, .4], [.5, .5], [.7, .3], [.4, .6],
                       [.5, .5], [.3, .7], [.55, .45], [.65, .35]])
    send, comb = solve(hidden, ti, tw, 4, 2)
    exp_s = torch.tensor([[4, 4], [4, 4]])
    assert (send == exp_s).all(), send.tolist()
    exp_c = torch.tensor([[1.4, 2.4], [4.5, 5.5], [7.3, 8.3], [8.2, 9.2],
                          [10.0, 11.0], [13.4, 14.4], [14.1, 15.1], [17.3, 18.3]])
    assert torch.allclose(comb, exp_c, atol=1e-5), comb.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(15):
        T = int(torch.randint(6, 12, (1,)))
        E = 4
        EPD = [1, 2, 4][int(torch.randint(0, 3, (1,)))]
        k = int(torch.randint(1, E + 1, (1,)))
        hidden = torch.randn(T, 3)
        ti = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
        tw = torch.rand(T, k)
        tw = tw / tw.sum(-1, keepdim=True)
        s1, c1 = solve(hidden, ti, tw, E, EPD)
        s2, c2 = _oracle(hidden, ti, tw, E, EPD)
        assert (s1 == s2).all(), (T, E, EPD, k)
        assert torch.allclose(c1, c2, atol=1e-5)


def check_send_total_equals_Tk(solve):
    torch.manual_seed(1)
    T, E, k = 10, 4, 2
    hidden = torch.randn(T, 3)
    ti = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
    tw = torch.rand(T, k)
    tw = tw / tw.sum(-1, keepdim=True)
    send, _ = solve(hidden, ti, tw, E, 2)
    assert int(send.sum()) == T * k, send.tolist()


def check_all_on_one_device_when_EPD_equals_E(solve):
    hidden = torch.randn(6, 2)
    ti = torch.stack([torch.randperm(4)[:2] for _ in range(6)])
    tw = torch.rand(6, 2)
    tw = tw / tw.sum(-1, keepdim=True)
    send, _ = solve(hidden, ti, tw, 4, 4)  # P=1
    assert send.shape == (1, 1)
    assert int(send[0, 0]) == 12


def check_weights_sum_to_one_preserved(solve):
    # 若 tw 每行和为 1 且 idx 恒为常数 e，则 combined[t] = hidden[t] + e
    hidden = torch.randn(4, 3)
    ti = torch.full((4, 2), 3, dtype=torch.long)  # 都选专家 3
    tw = torch.tensor([[0.5, 0.5]] * 4)
    _, comb = solve(hidden, ti, tw, 4, 1)
    assert torch.allclose(comb, hidden + 3.0, atol=1e-6), comb.tolist()


check_example.example = True
