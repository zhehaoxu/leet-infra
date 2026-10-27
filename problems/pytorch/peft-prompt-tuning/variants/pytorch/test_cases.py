import torch

POOL = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])   # (4,2)
TASK_IDS = torch.tensor([[0, 1], [3, 2]])                               # (2,2)
E = torch.tensor([[9.0, 9.0]])                                          # (1,2)


def _oracle(pool, task_ids, E):
    """独立重算：纯 python 双循环逐行拼接。"""
    T, L = task_ids.shape
    S0, d = E.shape
    out = []
    for t in range(T):
        block = []
        for row in task_ids[t].tolist():
            block.append([float(v) for v in pool[int(row)]])
        for i in range(S0):
            block.append([float(v) for v in E[i]])
        out.append(block)
    return torch.tensor(out, dtype=torch.float64).to(torch.float32), int(T * L * d)


def check_example(solve):
    X, n = solve(POOL, TASK_IDS, E)
    exp = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
                        [[0.0, 2.0], [2.0, 0.0], [9.0, 9.0]]])
    assert torch.allclose(X, exp, atol=1e-5, rtol=1e-5), f"got {X.tolist()}"
    assert n == 8, f"n_params 应为 8，got {n}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(81)
    P, d, T, L, S0 = 8, 6, 3, 4, 5
    pool = torch.randn(P, d)
    E = torch.randn(S0, d)
    task_ids = torch.stack([torch.randperm(P)[:L] for _ in range(T)])
    X, n = solve(pool, task_ids, E)
    exp_X, exp_n = _oracle(pool, task_ids, E)
    assert torch.allclose(X, exp_X, atol=1e-5, rtol=1e-5), "X 与独立重算不符"
    assert n == exp_n, f"n_params 应为 {exp_n}，got {n}"


def check_gather_indices(solve):
    torch.manual_seed(91)
    P, d, T, L, S0 = 10, 3, 4, 3, 2
    pool = torch.randn(P, d)
    E = torch.randn(S0, d)
    task_ids = torch.stack([torch.randperm(P)[:L] for _ in range(T)])
    X, _ = solve(pool, task_ids, E)
    for t in range(T):
        assert torch.equal(X[t, :L], pool[task_ids[t]]), (
            f"任务 {t} 的前缀行与 gather 结果不符")


def check_prepend_order(solve):
    torch.manual_seed(101)
    P, d, T, L, S0 = 6, 3, 2, 2, 4
    pool = torch.randn(P, d)
    E = torch.randn(S0, d)
    task_ids = torch.stack([torch.randperm(P)[:L] for _ in range(T)])
    X, _ = solve(pool, task_ids, E)
    assert torch.equal(X[:, L:], E.unsqueeze(0).expand(T, -1, -1)), (
        "X[t,L:] 应等于原嵌入 E（提示在前、嵌入在后）")


def check_param_account(solve):
    torch.manual_seed(111)
    d, T, L, S0 = 4, 3, 2, 2
    E = torch.randn(S0, d)
    task_ids = torch.tensor([[0, 1], [2, 3], [4, 5]])
    X1, n1 = solve(torch.randn(6, d), task_ids, E)
    X2, n2 = solve(torch.randn(9, d), task_ids, E)
    assert n1 == n2 == T * L * d, (
        f"n_params 应为 {T*L*d} 且与池大小无关，got {n1}/{n2}")
    assert X1.shape == (T, L + S0, d) and X2.shape == (T, L + S0, d)
