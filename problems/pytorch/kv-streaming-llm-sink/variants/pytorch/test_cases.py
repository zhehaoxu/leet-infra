import torch


def _oracle(k_len, n_sink, cr):
    """独立：先算 n_kept，再显式列出保留位集合。"""
    n_kept = int(k_len * (1 - cr))
    n_pruned = k_len - n_kept
    keep = set(list(range(n_sink)) + list(range(n_sink + n_pruned, k_len)))
    out = torch.zeros(k_len, dtype=torch.float32)
    for i in keep:
        out[i] = 1.0
    return out


def check_example(solve):
    out = solve(10, 2, 0.4)
    expect = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(out, expect), out


def check_matches_oracle(solve):
    for k_len in (8, 12, 20, 33, 64):
        for n_sink in (0, 1, 2, 4):
            for cr in (0.0, 0.1, 0.25, 0.4, 0.5, 0.75):
                if n_sink >= k_len:
                    continue
                got = solve(k_len, n_sink, cr)
                exp = _oracle(k_len, n_sink, cr)
                assert torch.allclose(got, exp), (k_len, n_sink, cr, got, exp)


def check_no_compression_keeps_all(solve):
    out = solve(6, 2, 0.0)
    assert torch.allclose(out, torch.ones(6)), out


def check_sink_and_window_boundaries(solve):
    # k_len=8, n_sink=4, cr=0.25 → n_pruned=2，剪 [4,6)
    out = solve(8, 4, 0.25)
    assert out.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], out


def check_pruned_count_matches(solve):
    # 打分数值中 0 的个数就是 n_pruned = k_len - int(k_len * (1-cr))
    for k_len, cr in [(10, 0.4), (13, 0.3), (20, 0.15), (50, 0.5)]:
        out = solve(k_len, 2, cr)
        expected_zeros = k_len - int(k_len * (1 - cr))
        assert int((out == 0).sum().item()) == expected_zeros, (k_len, cr, out)


check_example.example = True
