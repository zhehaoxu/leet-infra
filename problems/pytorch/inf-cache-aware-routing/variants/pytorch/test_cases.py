import random

import numpy as np
import torch


def _blocks(row):
    out = []
    for v in row:
        if v < 0:
            break
        out.append(int(v))
    return out


def _oracle(cache, queries, queue, alpha, beta):
    """独立 oracle：每步重建缓存集合，用 numpy argmin 取最小代价值（入参为纯 Python 列表）。"""
    sets = [set(v for v in row if v >= 0) for row in cache]
    depths = [int(v) for v in queue]
    chosen, hits = [], []
    total = 0.0

    for row in queries:
        blocks = _blocks(row)
        matched = []
        for s in sets:
            k = 0
            while k < len(blocks) and blocks[k] in s:
                k += 1
            matched.append(k)
        costs = alpha * (len(blocks) - np.array(matched, dtype=np.float64)) + beta * np.array(
            depths, dtype=np.float64
        )
        r = int(np.argmin(costs))
        chosen.append(r)
        hits.append(matched[r])
        total += float(costs[r])
        depths[r] += 1
        sets[r].update(blocks)

    return chosen, hits, depths, total


def check_example(solve):
    torch.manual_seed(0)
    chosen, hits, final_queue, total = solve(
        torch.tensor([[10, 11, 12], [10, 20, 21]]),
        torch.tensor([[10, 11, 5], [10, 20, 21]]),
        torch.tensor([0, 0]),
        1.0,
        1.0,
    )
    assert chosen.tolist() == [0, 1], f"副本选择应为 [0, 1]，got {chosen.tolist()}"
    assert hits.tolist() == [2, 3], f"命中块数应为 [2, 3]，got {hits.tolist()}"
    assert final_queue.tolist() == [1, 1], f"got {final_queue.tolist()}"
    assert abs(float(total) - 1.0) < 1e-6, f"总代价应为 1.0，got {float(total)}"


check_example.example = True


def check_queue_pressure_beats_cache(solve):
    """缓存完全一样时，队列更浅的副本胜出——粘性会被负载打破。"""
    torch.manual_seed(1)
    cache = torch.tensor([[1, 2, 3], [1, 2, 3]])
    queries = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    chosen, hits, final_queue, total = solve(cache, queries, torch.tensor([0, 0]), 1.0, 1.0)
    assert chosen.tolist() == [0, 1, 0], f"第三次两者队列一样深，并列取小副本号，got {chosen.tolist()}"
    assert hits.tolist() == [3, 3, 3], f"两个副本都全命中，got {hits.tolist()}"
    assert final_queue.tolist() == [2, 1], f"got {final_queue.tolist()}"
    assert abs(float(total) - 1.0) < 1e-6, f"只有第二次吃到队列惩罚，got {float(total)}"


def check_weight_extremes(solve):
    """alpha=0 退化为最少排队（按副本号轮转）；beta=0 退化为纯前缀命中。"""
    torch.manual_seed(2)
    queries = torch.tensor([[i] for i in range(1, 7)])
    empty = torch.full((3, 1), -1, dtype=torch.int64)
    chosen, hits, final_queue, total = solve(empty, queries, torch.zeros(3, dtype=torch.int64), 0.0, 1.0)
    assert chosen.tolist() == [0, 1, 2, 0, 1, 2], f"应按队列深度轮转，got {chosen.tolist()}"
    assert hits.tolist() == [0] * 6, "缓存为空不该有命中"
    assert final_queue.tolist() == [2, 2, 2], f"got {final_queue.tolist()}"
    assert abs(float(total) - 3.0) < 1e-6, f"后三次都落在队列深度 1 的副本上，代价应为 3，got {float(total)}"

    cache = torch.tensor([[1, -1], [1, 2]])
    chosen, hits, _, total = solve(
        cache, torch.tensor([[1, 2, 3]]), torch.zeros(2, dtype=torch.int64), 1.0, 0.0
    )
    assert chosen.tolist() == [1], f"应选命中更长的副本 1，got {chosen.tolist()}"
    assert hits.tolist() == [2] and abs(float(total) - 1.0) < 1e-6, f"got {hits.tolist()} / {float(total)}"


def check_random_against_oracle(solve):
    """随机副本缓存与请求流对拍 numpy oracle。"""
    rng = random.Random(9)
    for _ in range(30):
        n_rep = rng.randint(1, 3)
        width = rng.randint(1, 4)
        cache = [[rng.randint(0, 5) for _ in range(width)] for _ in range(n_rep)]
        n_req = rng.randint(1, 6)
        queries, lengths = [], []
        for _ in range(n_req):
            k = rng.randint(1, width)
            queries.append([rng.randint(0, 5) for _ in range(k)] + [-1] * (width - k))
            lengths.append(k)
        queue = [rng.randint(0, 3) for _ in range(n_rep)]
        alpha = rng.choice([0.0, 0.5, 1.0, 2.0])
        beta = rng.choice([0.0, 0.5, 1.0, 2.0])
        got = solve(
            torch.tensor(cache, dtype=torch.int64),
            torch.tensor(queries, dtype=torch.int64),
            torch.tensor(queue, dtype=torch.int64),
            alpha,
            beta,
        )
        want = _oracle(cache, queries, queue, alpha, beta)
        assert got[0].tolist() == want[0], f"chosen 不一致：{got[0].tolist()} vs {want[0]}"
        assert got[1].tolist() == want[1], f"hits 不一致：{got[1].tolist()} vs {want[1]}"
        assert got[2].tolist() == want[2], f"queue 不一致：{got[2].tolist()} vs {want[2]}"
        assert abs(float(got[3]) - want[3]) < 1e-5, f"代价不一致：{float(got[3])} vs {want[3]}"


def check_invalid_inputs(solve):
    c = torch.tensor([[1, 2]])
    q = torch.tensor([[1]])
    z = torch.zeros(1, dtype=torch.int64)
    for bad in (
        lambda: solve(torch.tensor([1, 2]), q, z, 1.0, 1.0),
        lambda: solve(c, torch.tensor([1]), z, 1.0, 1.0),
        lambda: solve(c, q, torch.zeros(2, dtype=torch.int64), 1.0, 1.0),
        lambda: solve(c, q, z, -1.0, 1.0),
        lambda: solve(c, q, z, 1.0, -0.5),
        lambda: solve(torch.zeros((0, 2), dtype=torch.int64), q, torch.zeros(0, dtype=torch.int64), 1.0, 1.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
