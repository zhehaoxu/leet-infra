import torch

ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
INV = (0, 4, 1, 5, 2, 6, 3, 7)


def solve(qweight, qzeros, scales, perm, K, N):
    if K % 16 != 0 or N % 16 != 0:
        raise ValueError("K and N must be multiples of 16")
    n_row = K // 16
    n_col = N // 16
    q = qweight.to(torch.int64)
    codes = torch.empty(K, N, dtype=torch.int64)
    for h in range(2):
        for b in range(8):
            local = 8 * h + INV[b]
            cols = 16 * torch.arange(n_col) + local
            vals = (q[:, :, :, h] >> (4 * b)) & 0xF
            codes[:, cols] = vals.permute(0, 2, 1).reshape(K, n_col)

    z = qzeros.to(torch.int64)
    zero_storage = torch.empty(N, dtype=torch.int64)
    for h in range(2):
        for b in range(8):
            local = 8 * h + INV[b]
            zero_storage[16 * torch.arange(n_col) + local] = (z[:, h] >> (4 * b)) & 0xF

    perm_idx = torch.tensor(list(perm), dtype=torch.int64)
    centered = torch.empty(K, N, dtype=torch.float64)
    centered[:, perm_idx] = (codes - zero_storage).to(torch.float64)
    W = centered * scales.to(torch.float64)[None, :]
    return W.to(torch.float32)
