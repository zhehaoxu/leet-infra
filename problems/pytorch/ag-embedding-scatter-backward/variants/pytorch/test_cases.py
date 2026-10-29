import torch
import torch.nn.functional as F

# 示例：num_embeddings=4、dim=2，idx=[1,1,0] ⇒ 第 1 行是两次命中之和 [2+6, 4+8]
EXAMPLE_GRAD = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]
EXAMPLE_IDX = [1, 1, 0]
EXAMPLE_OUT = [[10.0, 12.0], [8.0, 12.0], [0.0, 0.0], [0.0, 0.0]]
# 同数据 + scale_grad_by_freq：id 1 出现 2 次 ⇒ 每份先除 2 ⇒ 第 1 行 [1+3, 2+4]
EXAMPLE_FREQ = [[10.0, 12.0], [4.0, 6.0], [0.0, 0.0], [0.0, 0.0]]


def check_example(solve):
    out = solve(torch.tensor(EXAMPLE_GRAD), torch.tensor(EXAMPLE_IDX, dtype=torch.long), 4)
    assert out.shape == (4, 2)
    assert torch.allclose(out, torch.tensor(EXAMPLE_OUT), atol=1e-6)


check_example.example = True


def check_matches_autograd_duplicates(solve):
    torch.manual_seed(901)
    num, dim, n = 7, 4, 20
    w = torch.randn(num, dim, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, num, (n,), dtype=torch.long)
    dOut = torch.randn(n, dim, dtype=torch.float64)
    (ref,) = torch.autograd.grad((F.embedding(idx, w) * dOut).sum(), w)
    got = solve(dOut, idx, num)
    assert got.shape == ref.shape and got.dtype == dOut.dtype
    assert torch.allclose(got.double(), ref, atol=1e-9)
    # 重复计数与逐行累加的独立口径
    manual = torch.zeros(num, dim, dtype=torch.float64)
    for i, t in enumerate(idx.tolist()):
        manual[t] += dOut[i]
    assert torch.allclose(got.double(), manual, atol=1e-9)


def check_padding_idx(solve):
    torch.manual_seed(902)
    num, dim, n = 6, 3, 15
    pad = 0
    w = torch.randn(num, dim, dtype=torch.float64, requires_grad=True)
    w.data[pad] = 0.0
    idx = torch.randint(0, num, (n,), dtype=torch.long)
    dOut = torch.randn(n, dim, dtype=torch.float64)
    (ref,) = torch.autograd.grad((F.embedding(idx, w, padding_idx=pad) * dOut).sum(), w)
    got = solve(dOut, idx, num, padding_idx=pad)
    assert torch.allclose(got.double(), ref, atol=1e-9)
    assert torch.count_nonzero(got[pad]) == 0
    # 命中全在 padding 行时整块梯度为 0
    only = torch.full((4,), pad, dtype=torch.long)
    z = solve(torch.ones(4, dim, dtype=torch.float64), only, num, padding_idx=pad)
    assert torch.count_nonzero(z) == 0


def check_scale_grad_by_freq(solve):
    torch.manual_seed(903)
    num, dim, n = 5, 3, 24
    w = torch.randn(num, dim, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, num, (n,), dtype=torch.long)
    dOut = torch.randn(n, dim, dtype=torch.float64)
    (ref,) = torch.autograd.grad(
        (F.embedding(idx, w, scale_grad_by_freq=True) * dOut).sum(), w)
    got = solve(dOut, idx, num, scale_grad_by_freq=True)
    assert torch.allclose(got.double(), ref, atol=1e-9)
    # 手算口径：先按出现次数摊薄再累加
    cnt = torch.bincount(idx, minlength=num).double()
    manual = torch.zeros(num, dim, dtype=torch.float64)
    for i, t in enumerate(idx.tolist()):
        manual[t] += dOut[i] / cnt[t]
    assert torch.allclose(got.double(), manual, atol=1e-9)
    # 示例数据摊薄后第 1 行是 [4,6]
    out = solve(torch.tensor(EXAMPLE_GRAD), torch.tensor(EXAMPLE_IDX, dtype=torch.long),
                4, scale_grad_by_freq=True)
    assert torch.allclose(out, torch.tensor(EXAMPLE_FREQ), atol=1e-6)


def check_sparse_matches_dense(solve):
    torch.manual_seed(904)
    num, dim, n = 8, 3, 12
    w = torch.randn(num, dim, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, num, (n,), dtype=torch.long)
    dOut = torch.randn(n, dim, dtype=torch.float64)
    (ref,) = torch.autograd.grad((F.embedding(idx, w) * dOut).sum(), w)
    dense = solve(dOut, idx, num)
    sp = solve(dOut, idx, num, sparse=True)
    assert sp.is_sparse
    assert tuple(sp.shape) == (num, dim)
    assert torch.allclose(sp.to_dense(), dense, atol=1e-9)
    assert torch.allclose(sp.to_dense().double(), ref, atol=1e-9)
    # 稀疏表示已合并：同一 (row, col) 不重复出现
    co = sp.coalesce()
    ind = co.indices()
    keys = (ind[0] * dim + ind[1]).tolist()
    assert len(keys) == len(set(keys))


def check_invalid_args(solve):
    g = torch.randn(3, 2)
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    bad = [
        (g, idx, 2),          # 索引越界（有 2）
        (g, idx, 0),          # num_embeddings 非正
        (g, torch.tensor([0, -1, 2], dtype=torch.long), 5),   # 负索引
        (g.unsqueeze(0), idx, 5),                              # grad_out 非二维
        (g, idx.reshape(1, 3), 5),                             # idx 非一维
        (torch.randn(4, 2), idx, 5),                           # 行数与 idx 不符
    ]
    for gg, ii, num in bad:
        try:
            solve(gg, ii, num)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
