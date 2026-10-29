"""独立 oracle：prefill 队列（deque）+ decode 列表的另一套写法，与参考实现不共享流程。"""

import random
from collections import deque


def _oracle(arrivals, prompt_lens, gen_lens, chunk, max_batch):
    n = len(arrivals)
    pending = deque(sorted(range(n), key=lambda i: (arrivals[i], i)))
    prefilling = deque()
    decoding = []
    pref_left = list(prompt_lens)
    left = list(gen_lens)
    first = [-1] * n
    finish = [-1] * n
    prefill_steps = 0
    decode_steps = 0
    t = 0

    while any(f < 0 for f in finish):
        while (
            len(prefilling) + len(decoding) < max_batch
            and pending
            and arrivals[pending[0]] <= t
        ):
            prefilling.append(pending.popleft())

        if prefilling:
            i = prefilling[0]
            pref_left[i] -= chunk
            prefill_steps += 1
            if pref_left[i] <= 0:
                prefilling.popleft()
                decoding.append(i)
            t += 1
            continue

        if decoding:
            decode_steps += 1
            still = []
            for i in decoding:
                left[i] -= 1
                if left[i] == gen_lens[i] - 1:
                    first[i] = t
                if left[i] == 0:
                    finish[i] = t
                else:
                    still.append(i)
            decoding = still
            t += 1
            continue

        assert pending, "no work left but requests remain unfinished"
        t = max(t, arrivals[pending[0]])

    ttft = [first[i] - arrivals[i] for i in range(n)]
    decode_span = [finish[i] - first[i] for i in range(n)]
    return ttft, decode_span, finish, prefill_steps, decode_steps


def check_example(solve):
    ttft, span, finish, prefill_steps, decode_steps = solve(
        [0, 0, 2], [8, 4, 4], [2, 3, 1], 4, 2
    )
    assert ttft == [3, 3, 4], f"ttft 应为 [3, 3, 4]，got {ttft}"
    assert span == [1, 3, 0], f"decode_span 应为 [1, 3, 0]，got {span}"
    assert finish == [4, 6, 6], f"finish 应为 [4, 6, 6]，got {finish}"
    assert prefill_steps == 4, f"prefill 步数应为 4，got {prefill_steps}"
    assert decode_steps == 3, f"decode 步数应为 3，got {decode_steps}"


check_example.example = True


def check_single_request(solve):
    """单请求无争用：chunk 正好吃掉 prompt，1 个 prefill 步 + gen_len 个 decode 步。"""
    ttft, span, finish, prefill_steps, decode_steps = solve([5], [4], [3], 4, 1)
    assert ttft == [1], f"首个 token 在 prefill 完成的下一步，got {ttft}"
    assert span == [2], f"3 个 token 跨 2 步，got {span}"
    assert finish == [8], f"got {finish}"
    assert (prefill_steps, decode_steps) == (1, 3), (prefill_steps, decode_steps)


def check_prefill_steals_decode(solve):
    """在跑请求的 decode 被后来者的 prefill 挤掉：ITL 从 1 步涨到 7/3 步。"""
    ttft, span, finish, prefill_steps, decode_steps = solve([0, 2], [1, 8], [4, 1], 2, 2)
    assert ttft == [1, 4], f"got {ttft}"
    assert span == [7, 0], f"r0 的 4 个 token 被 4 个 prefill 步拉开，got {span}"
    assert finish == [8, 6], f"got {finish}"
    assert (prefill_steps, decode_steps) == (5, 4), (prefill_steps, decode_steps)


def check_batch_waves(solve):
    """max_batch=2、四个同时到达：完成的名额下一步才能用，于是分成两波。"""
    ttft, span, finish, prefill_steps, decode_steps = solve(
        [0, 0, 0, 0], [2, 2, 2, 2], [2, 2, 2, 2], 4, 2
    )
    assert ttft == [2, 2, 6, 6], f"got {ttft}"
    assert span == [1, 1, 1, 1], f"got {span}"
    assert finish == [3, 3, 7, 7], f"第 3、4 个请求要到第 4 步才准入，got {finish}"
    assert (prefill_steps, decode_steps) == (4, 4), (prefill_steps, decode_steps)


def check_admission_one_slot(solve):
    """max_batch=1：请求严格串行，每个新请求比上一个晚 2 步首字。"""
    ttft, span, finish, prefill_steps, decode_steps = solve(
        [0, 0, 0], [1, 1, 1], [1, 1, 1], 1, 1
    )
    assert ttft == [1, 3, 5], f"got {ttft}"
    assert span == [0, 0, 0], f"got {span}"
    assert finish == [1, 3, 5], f"got {finish}"
    assert (prefill_steps, decode_steps) == (3, 3), (prefill_steps, decode_steps)


def check_idle_gap_not_counted(solve):
    """空转到下一个到达步不算步数，也不污染 prefill/decode 计数。"""
    ttft, span, finish, prefill_steps, decode_steps = solve([10], [4], [2], 4, 1)
    assert ttft == [1], f"got {ttft}"
    assert span == [1], f"got {span}"
    assert finish == [12], f"finish 是绝对步号，got {finish}"
    assert (prefill_steps, decode_steps) == (1, 2), (prefill_steps, decode_steps)


def check_out_of_order_arrivals(solve):
    """到达顺序乱序：准入按 (arrival, 下标)，不是按数组顺序。"""
    ttft, span, finish, prefill_steps, decode_steps = solve([3, 0], [2, 2], [1, 1], 2, 2)
    assert ttft == [1, 1], f"got {ttft}"
    assert span == [0, 0], f"got {span}"
    assert finish == [4, 1], f"下标 1 先到先跑，got {finish}"
    assert (prefill_steps, decode_steps) == (2, 2), (prefill_steps, decode_steps)


def check_random_against_oracle(solve):
    """随机时间线与独立 oracle 全等，并核对基本不变量。"""
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randint(1, 5)
        arrivals = [rng.randint(0, 6) for _ in range(n)]
        prompts = [rng.randint(1, 9) for _ in range(n)]
        gens = [rng.randint(1, 4) for _ in range(n)]
        chunk = rng.randint(1, 5)
        max_batch = rng.randint(1, 3)
        got = solve(arrivals, prompts, gens, chunk, max_batch)
        want = _oracle(arrivals, prompts, gens, chunk, max_batch)
        tag = (arrivals, prompts, gens, chunk, max_batch)
        assert got == want, f"{tag}: {got} vs {want}"
        ttft, span, finish, prefill_steps, decode_steps = got
        for i in range(n):
            prefill_ticks = -(-prompts[i] // chunk)
            assert ttft[i] >= prefill_ticks, f"{tag} 请求 {i} 的 TTFT 小于 prefill 时长"
            assert span[i] >= gens[i] - 1, f"{tag} 请求 {i} 的 span 小于 token 间隔数"
            assert finish[i] == arrivals[i] + ttft[i] + span[i], f"{tag} 请求 {i} 时间账不平"
        assert prefill_steps == sum(-(-p // chunk) for p in prompts), f"{tag} prefill 步数应等于各请求分块数之和"
        assert max(gens) <= decode_steps <= sum(gens), f"{tag} decode 步数越界"
