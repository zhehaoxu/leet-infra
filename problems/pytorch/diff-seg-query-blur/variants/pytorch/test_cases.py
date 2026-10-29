import math

import torch


def _reflect_index(i, n):
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        else:
            i = 2 * (n - 1) - i
    return i


def _kernel_size(sigma, grid):
    raw = math.ceil(6.0 * sigma)
    return min(raw + 1 - raw % 2, grid - (grid % 2 - 1))


def _blur_plane(plane, sigma, size):
    """对单个 n x n 特征图做 size x size 的可分离高斯模糊（反射填充）。"""
    grid = len(plane)
    half = (size - 1) / 2.0
    axis = [-half + 2.0 * half * k / (size - 1) for k in range(size)] if size > 1 else [0.0]
    pdf = [math.exp(-0.5 * (x / sigma) ** 2) for x in axis]
    total = sum(pdf)
    pdf = [p / total for p in pdf]
    pad = size // 2
    out = []
    for i in range(grid):
        for j in range(grid):
            acc = 0.0
            for a in range(size):
                ri = _reflect_index(i - pad + a, grid)
                for b in range(size):
                    rj = _reflect_index(j - pad + b, grid)
                    acc += plane[ri][rj] * pdf[a] * pdf[b]
            out.append(acc)
    return out


def _oracle(query, sigma, threshold):
    """独立 oracle：反射索引手写，卷积用四重 python 循环。"""
    batch, seq_len, dim = query.shape
    grid = int(math.sqrt(seq_len))
    count = grid * grid
    out = query.tolist()
    for b in range(batch):
        for d in range(dim):
            plane = [[query[b, i * grid + j, d].item() for j in range(grid)] for i in range(grid)]
            if sigma > threshold:
                smoothed = _blur_plane(plane, sigma, _kernel_size(sigma, grid))
            else:
                mean = sum(sum(row) for row in plane) / count
                smoothed = [mean] * count
            for idx in range(count):
                out[b][idx][d] = smoothed[idx]
    return torch.tensor(out, dtype=torch.float64)


def _grid_query(rows, cols, dim, trailing=0):
    total = rows * cols + trailing
    q = torch.zeros(1, total, dim, dtype=torch.float64)
    for t in range(total):
        for d in range(dim):
            q[0, t, d] = 10.0 * d + t
    return q


def check_example(solve):
    q = torch.zeros(1, 10, 2, dtype=torch.float64)
    for t in range(10):
        q[0, t, 0] = float(t)
        q[0, t, 1] = 10.0 + t

    # sigma=1.0 不大于阈值 9999：整块均值；第 9 个 token 是尾巴，原样保留
    mean = solve(q, 1.0)
    assert mean.dtype == torch.float32 and mean.shape == q.shape
    for t in range(9):
        assert [round(float(v), 6) for v in mean[0, t].tolist()] == [4.0, 14.0], f"token {t} 应被均值替换"
    assert [round(float(v), 6) for v in mean[0, 9].tolist()] == [9.0, 19.0], "尾部 token 不应被改动"

    # sigma=50000 大于阈值：3x3 反射填充的近似均匀模糊
    blurred = solve(q, 50000.0)
    assert [[round(float(v), 6) for v in row] for row in blurred[0, :9].tolist()] == [
        [2.666667, 12.666667],
        [3.0, 13.0],
        [3.333333, 13.333333],
        [3.666667, 13.666667],
        [4.0, 14.0],
        [4.333333, 14.333333],
        [4.666667, 14.666667],
        [5.0, 15.0],
        [5.333333, 15.333333],
    ], blurred.tolist()
    assert [round(float(v), 6) for v in blurred[0, 9].tolist()] == [9.0, 19.0]


check_example.example = True


def check_mean_branch(solve):
    torch.manual_seed(0)
    q = torch.randn(2, 9, 3, dtype=torch.float64)
    got = solve(q, 5.0).to(torch.float64)
    want = q.mean(dim=1, keepdim=True).expand_as(q)
    assert torch.allclose(got, want, atol=1e-6), "均值分支应把整块压成同一个值"

    # 每个通道独立求均值：不同通道的均值不同
    assert not torch.allclose(got[0, 0, :], got[1, 0, :], atol=1e-3)
    assert torch.allclose(got, want, atol=1e-6)


def check_threshold_is_strict(solve):
    torch.manual_seed(1)
    q = _grid_query(3, 3, 2)

    # sigma 恰好等于阈值：走均值分支（严格大于才算无限模糊）
    at = solve(q, 7.5, blur_threshold_inf=7.5)
    assert torch.allclose(at, q.mean(dim=1, keepdim=True).expand_as(q).to(torch.float32), atol=1e-6)
    # 阈值更低时同一步改走卷积分支，结果不同
    over = solve(q, 7.5, blur_threshold_inf=7.4).to(torch.float64)
    assert not torch.allclose(over, at.to(torch.float64), atol=1e-3)


def check_kernel_clamping(solve):
    # n=6（偶数），sigma=1.5 -> 原始核长 9，被夹到 n+1=7
    torch.manual_seed(2)
    q = _grid_query(6, 6, 2)
    got = solve(q, 1.5, blur_threshold_inf=1.0).to(torch.float64)
    want = _oracle(q, 1.5, 1.0)
    assert torch.allclose(got, want, atol=1e-5), "偶数网格的核长上限应为 n+1"

    # 原始核长 9 未被夹取时会给出另一个结果，说明上限确实生效
    plane = [[q[0, i * 6 + j, 0].item() for j in range(6)] for i in range(6)]
    unclamped = torch.tensor(_blur_plane(plane, 1.5, 9), dtype=torch.float64).reshape(6, 6)
    assert not torch.allclose(got[0, :36, 0].reshape(6, 6).to(torch.float64), unclamped, atol=1e-4)

    # n=3（奇数），核长上限就是 n=3
    q3 = _grid_query(3, 3, 2)
    got3 = solve(q3, 40.0, blur_threshold_inf=1.0).to(torch.float64)
    assert torch.allclose(got3, _oracle(q3, 40.0, 1.0), atol=1e-5)


def check_trailing_tokens(solve):
    # 10 个 token -> n=3，9 个参与平滑，最后 1 个原样保留
    q = _grid_query(3, 3, 3, trailing=1)
    got = solve(q, 50000.0)
    assert torch.equal(got[:, 9:, :], q[:, 9:, :].to(torch.float32)), "尾部 token 必须逐位不变"
    # token 数不足一个方格时（S=1, n=1）也成立
    single = torch.tensor([[[2.0, 5.0]]])
    assert torch.equal(solve(single, 50000.0), single)


def check_against_oracle(solve):
    settings = [
        (3, 2, 1.0, 9999.0),
        (3, 2, 50000.0, 9999.0),
        (4, 3, 1.5, 1.0),
        (5, 1, 2.0, 1.0),
        (2, 1, 3.0, 2.5),
        (6, 2, 0.9, 0.5),
    ]
    for grid, dim, sigma, threshold in settings:
        q = _grid_query(grid, grid, dim, trailing=0)
        got = solve(q, sigma, blur_threshold_inf=threshold).to(torch.float64)
        want = _oracle(q, sigma, threshold)
        assert torch.allclose(got, want, atol=1e-5), f"grid={grid}, dim={dim}, sigma={sigma} 与 oracle 不一致"

    # 非方形 token 数：多出来的尾巴不参与，也不被改写
    torch.manual_seed(3)
    q = torch.randn(2, 10, 2, dtype=torch.float64)
    got = solve(q, 1.0).to(torch.float64)
    want = _oracle(q, 1.0, 9999.0)
    assert torch.allclose(got, want, atol=1e-6)


def check_no_mutation(solve):
    torch.manual_seed(4)
    q = torch.randn(2, 9, 3, dtype=torch.float64)
    backup = q.clone()
    solve(q, 1.0)
    solve(q, 50000.0)
    assert torch.equal(q, backup), "solve 不应就地修改输入"


def check_invalid(solve):
    q = torch.zeros(2, 4, 3)
    for bad in (
        lambda: solve([[0.0, 1.0]], 1.0),
        lambda: solve(torch.zeros(4, 3), 1.0),
        lambda: solve(torch.zeros(2, 2, 4, 3), 1.0),
        lambda: solve(q, -1.0),
        lambda: solve(q, float("nan")),
        lambda: solve(q, float("inf")),
        lambda: solve(q, "1.0"),
        lambda: solve(q, True),
        lambda: solve(q, 1.0, blur_threshold_inf=-2.0),
        lambda: solve(torch.zeros(1, 0, 3), 1.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
