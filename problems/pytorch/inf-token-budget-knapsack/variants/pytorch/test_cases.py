"""独立 oracle：枚举全部 2^n 个子集取最优，与参考实现的容量 DP 不共享思路。"""

import random


def _oracle(prompt_lens, values, budget, decode_count):
    remaining = budget - decode_count
    n = len(prompt_lens)
    best = None
    for mask in range(1 << n):
        idx = tuple(i for i in range(n) if mask >> i & 1)
        cost = sum(prompt_lens[i] for i in idx)
        if cost > remaining:
            continue
        val = sum(values[i] for i in idx)
        key = (-val, cost, idx)
        if best is None or key < best[0]:
            best = (key, cost, val, idx)
    _, cost_used, total_value, idx = best
    served = list(idx)
    chosen = set(served)
    leftover = remaining - cost_used
    chunk_index, chunk_tokens = -1, 0
    if leftover > 0:
        for i in range(n):
            if i not in chosen:
                chunk_index = i
                chunk_tokens = min(leftover, prompt_lens[i])
                break
    return served, total_value, chunk_index, chunk_tokens, leftover - chunk_tokens


def check_example(solve):
    got = solve([5, 3, 8, 3], [4, 2, 9, 1], 10, 2)
    assert got == ([2], 9, -1, 0, 0), f"高权重的大请求整段胜出，got {got}"


check_example.example = True


def check_leftover_goes_to_first_unserved(solve):
    """decode 吃掉 4 个名额后只剩 2 个：谁都装不下，余量整块给 FCFS 最靠前的请求。"""
    got = solve([4, 4], [3, 3], 6, 4)
    assert got == ([], 0, 0, 2, 0), f"got {got}"


def check_zero_value_never_served(solve):
    """权重为 0 的请求入选只会浪费名额，永远不进 S；它的 prompt 比余量大时吃不完余量。"""
    got = solve([1, 2, 6], [10, 0, 1], 10, 0)
    assert got == ([0, 2], 11, 1, 2, 1), f"got {got}"


def check_tie_prefers_smaller_cost(solve):
    """权重并列时选占用更小的子集，省下来的名额还能继续分块。"""
    got = solve([2, 4], [5, 5], 4, 0)
    assert got == ([0], 5, 1, 2, 0), f"got {got}"


def check_tie_prefers_lexicographic(solve):
    """权重与总占用都并列时，选升序下标列表字典序更小的集合。"""
    got = solve([3, 3, 3], [2, 2, 2], 6, 0)
    assert got == ([0, 1], 4, -1, 0, 0), f"正好的装满时没有余量可分块，got {got}"


def check_exact_fit(solve):
    """正好装满：没有余量，也就没有分块。"""
    got = solve([2, 3], [4, 6], 5, 0)
    assert got == ([0, 1], 10, -1, 0, 0), f"got {got}"


def check_decode_uses_whole_budget(solve):
    """budget 全被 decode 吃掉：remaining=0，什么都不选也不分块。"""
    got = solve([1, 2], [5, 5], 3, 3)
    assert got == ([], 0, -1, 0, 0), f"got {got}"


def check_none_fit(solve):
    """所有 prompt 都超过预算：全部落选，余量给第一个请求。"""
    got = solve([9, 9], [5, 5], 5, 0)
    assert got == ([], 0, 0, 5, 0), f"got {got}"


def check_all_served_leftover_unused(solve):
    """全部入选后仍有剩余名额：没有请求可分块，余量记为 unused。"""
    got = solve([2, 5], [10, 1], 8, 0)
    assert got == ([0, 1], 11, -1, 0, 1), f"got {got}"


def check_empty_queue(solve):
    got = solve([], [], 7, 2)
    assert got == ([], 0, -1, 0, 5), f"got {got}"


def check_random_against_oracle(solve):
    rng = random.Random(4861)
    for _ in range(200):
        n = rng.randint(0, 8)
        prompts = [rng.randint(1, 9) for _ in range(n)]
        values = [rng.randint(0, 5) for _ in range(n)]
        decode = rng.randint(0, 4)
        budget = decode + rng.randint(0, 18)
        got = solve(prompts, values, budget, decode)
        want = _oracle(prompts, values, budget, decode)
        tag = (prompts, values, budget, decode)
        assert got == want, f"{tag}: {got} vs {want}"

        served, total_value, chunk_index, chunk_tokens, unused = got
        remaining = budget - decode
        assert served == sorted(set(served)), f"{tag} served 必须是升序去重的下标"
        assert sum(prompts[i] for i in served) <= remaining, f"{tag} 入选集合超预算"
        assert total_value == sum(values[i] for i in served), f"{tag} 权重和不对"
        assert unused >= 0, f"{tag} unused 不能为负"
        if chunk_index >= 0:
            assert chunk_index not in served, f"{tag} 分块落在已入选的请求上"
            assert 0 < chunk_tokens <= min(prompts[chunk_index], remaining), f"{tag} 分块大小越界"
            assert chunk_index == min(i for i in range(n) if i not in set(served)), (
                f"{tag} 分块必须给下标最小的未入选请求"
            )
            assert chunk_tokens + unused == remaining - sum(prompts[i] for i in served), (
                f"{tag} 余量账不平"
            )
