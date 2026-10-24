import math

import torch


def _oracle(keys, indices, inv_freq):
    """独立纯 Python：逐 (head, 新位置) 手工算 gather + 重旋转，逐元素 cos/sin。"""
    H, L, D = keys.shape
    half = D // 2
    out = []
    for h in range(H):
        rows = []
        for i in range(indices.shape[1]):
            p = int(indices[h][i])
            k = [float(keys[h][p][t]) for t in range(D)]
            delta = i - p
            row = []
            for t in range(D):
                f = delta * float(inv_freq[t % half])
                c = math.cos(f)
                s = math.sin(f)
                rh = -k[t + half] if t < half else k[t - half]
                row.append(k[t] * c + rh * s)
            rows.append(row)
        out.append(rows)
    return torch.tensor(out, dtype=torch.float32)


def check_example(solve):
    keys = torch.tensor([[[1.0, 0.5, 0.2, 0.8],
                          [0.3, 0.7, 0.1, 0.4],
                          [0.9, 0.2, 0.6, 0.5],
                          [0.4, 0.4, 0.4, 0.4],
                          [0.7, 0.1, 0.3, 0.9]]])
    indices = torch.tensor([[0, 2, 4]])
    inv_freq = torch.tensor([1.0, 0.1])
    out = solve(keys, indices, inv_freq)
    exp = torch.tensor([[[1.0, 0.5, 0.2, 0.8],
                         [0.9912, 0.2489, -0.4331, 0.4775],
                         [-0.0185, 0.2768, -0.7614, 0.8622]]])
    assert torch.allclose(out, exp, atol=1e-4), out


def check_matches_oracle(solve):
    torch.manual_seed(3)
    for _ in range(30):
        H = int(torch.randint(1, 4, (1,)))
        L = int(torch.randint(2, 9, (1,)))
        D = 2 * int(torch.randint(1, 4, (1,)))  # 偶数
        keys = torch.randn(H, L, D)
        inv_freq = 10.0 ** -torch.arange(D // 2, dtype=torch.float32)
        n_kept = int(torch.randint(1, L + 1, (1,)))
        indices = []
        for _h in range(H):
            idx = torch.randperm(L)[:n_kept].sort().values
            indices.append(idx.tolist())
        indices = torch.tensor(indices, dtype=torch.long)
        got = solve(keys, indices, inv_freq)
        exp = _oracle(keys, indices, inv_freq)
        assert torch.allclose(got, exp, atol=1e-4), (H, L, D, got, exp)


def check_first_kept_unchanged_when_pos0(solve):
    # 若某 head 保留了位置 0 作为新位置 0（delta=0），该行应原样不变
    keys = torch.randn(1, 5, 4)
    indices = torch.tensor([[0, 3]])
    inv_freq = torch.tensor([0.5, 0.1])
    out = solve(keys, indices, inv_freq)
    assert torch.allclose(out[0, 0], keys[0, 0], atol=1e-6), out


def check_identity_when_contiguous(solve):
    # 保留连续位置 [0,1,...,L-1]：delta 全 0 → 完全不变
    H, L, D = 2, 5, 6
    keys = torch.randn(H, L, D)
    indices = torch.arange(L).unsqueeze(0).expand(H, L).contiguous()
    inv_freq = 1.0 / (100 ** torch.arange(D // 2, dtype=torch.float32))
    out = solve(keys, indices, inv_freq)
    assert torch.allclose(out, keys, atol=1e-5), out


check_example.example = True
