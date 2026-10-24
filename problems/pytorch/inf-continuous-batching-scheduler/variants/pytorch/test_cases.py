import torch


def _oracle(requests, max_batch_size):
    """独立：以「时间片轮转 + 严格 step 循环」重新模拟。"""
    N = requests.shape[0]
    finish = [-1] * N
    arrivals = [int(requests[i, 0]) for i in range(N)]
    gens = [int(requests[i, 1]) for i in range(N)]
    rem = [0] * N  # 剩余步
    running = set()
    waiting = set(range(N))
    step = 0
    while waiting or running:
        # admit 按 arrival 升序、id 升序
        admitted_now = sorted([i for i in waiting if arrivals[i] <= step],
                               key=lambda i: (arrivals[i], i))
        for i in admitted_now:
            if len(running) >= max_batch_size:
                break
            running.add(i)
            waiting.remove(i)
            rem[i] = gens[i] + 1
        # tick
        done_now = []
        for i in running:
            rem[i] -= 1
            if rem[i] == 0:
                done_now.append(i)
        for i in done_now:
            finish[i] = step
            running.remove(i)
        step += 1
    return torch.tensor(finish)


def check_example(solve):
    requests = torch.tensor([[0, 3], [0, 5], [1, 2], [2, 4], [3, 1]])
    fin = solve(requests, 2)
    assert fin.tolist() == [3, 5, 6, 10, 8], fin.tolist()


def check_single_request(solve):
    requests = torch.tensor([[0, 2]])
    fin = solve(requests, 4)
    assert fin.tolist() == [2], fin.tolist()


def check_no_concurrency_matches_serial(solve):
    # max_batch_size=1 → 严格串行
    requests = torch.tensor([[0, 1], [0, 1], [0, 1]])
    fin = solve(requests, 1)
    # 每请求占 2 step（1 prefill + 1 decode）
    assert fin.tolist() == [1, 3, 5], fin.tolist()


def check_matches_oracle(solve):
    torch.manual_seed(0)
    for _ in range(20):
        N = int(torch.randint(3, 9, (1,)))
        arr = sorted(int(torch.randint(0, 10, (1,))) for _ in range(N))
        gens = [int(torch.randint(1, 6, (1,))) for _ in range(N)]
        req = torch.tensor(list(zip(arr, gens)))
        mbs = int(torch.randint(1, 4, (1,)))
        got = solve(req, mbs)
        exp = _oracle(req, mbs)
        assert got.tolist() == exp.tolist(), (req.tolist(), mbs, got.tolist())


def check_all_finite(solve):
    torch.manual_seed(1)
    req = torch.tensor([[0, 3], [2, 1], [5, 4]])
    fin = solve(req, 2)
    assert (fin >= 0).all()


check_example.example = True
