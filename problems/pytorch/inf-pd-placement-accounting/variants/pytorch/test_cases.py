"""独立 oracle：把副本状态放进字典、候选先收集再排序，与参考实现的「边扫边比」写法不同。"""

import random


def _cdiv(x, y):
    q, r = divmod(x, y)
    return q + (1 if r else 0)


def _oracle(prompt_lens, gen_lens, kv_bytes_per_token, caps, bws, tps, prefill_tps):
    replicas = [
        {"cap": caps[d], "bw": bws[d], "tp": tps[d], "free": 0, "kv": 0, "wire": 0}
        for d in range(len(caps))
    ]
    placement = []
    ttft = []
    rejected = 0
    seen_prompt = 0

    for i in range(len(prompt_lens)):
        seen_prompt += prompt_lens[i]
        ready = _cdiv(seen_prompt, prefill_tps)
        need = (prompt_lens[i] + gen_lens[i]) * kv_bytes_per_token

        options = []
        for d, r in enumerate(replicas):
            if r["kv"] + need > r["cap"]:
                continue
            moved = need + _cdiv(need * (r["tp"] - 1), r["tp"])
            begin = ready if ready > r["free"] else r["free"]
            options.append((begin + _cdiv(moved, r["bw"]), d, moved))

        if not options:
            placement.append(-1)
            ttft.append(-1)
            rejected += 1
            continue

        options.sort(key=lambda o: (o[0], o[1]))
        finish, d, moved = options[0]
        r = replicas[d]
        r["kv"] += need
        r["wire"] += moved
        r["free"] = finish
        placement.append(d)
        ttft.append(finish)

    accepted = [t for t in ttft if t >= 0]
    return (
        placement,
        ttft,
        [r["wire"] for r in replicas],
        [r["kv"] for r in replicas],
        rejected,
        max(accepted) if accepted else 0,
    )


def check_example(solve):
    """容量、链路排队与 TP 环形复制同时出现：第 3 个请求被挤到 tp=2 的副本。"""
    got = solve([20, 30, 100, 10], [30, 10, 100, 5], 4, [1000, 1000], [64, 32], [1, 2], 10)
    assert got[0] == [0, 0, 1, 0], f"第 3 个请求 360+800=1160>1000，只能去副本 1，got {got[0]}"
    assert got[1] == [6, 9, 53, 17], f"got {got[1]}"
    assert got[2] == [420, 1200], f"副本 1 的 800 字节还要加 TP=2 的环形份额 400，got {got[2]}"
    assert got[3] == [420, 800], f"got {got[3]}"
    assert got[4] == 0 and got[5] == 53, f"got {got[4:]}"


check_example.example = True


def check_ring_share(solve):
    """tp=1 只搬 handoff；tp=4 还要搬 (C-1)/C 的环形 all-gather 份额。"""
    one = solve([10], [10], 4, [10 ** 6], [64], [1], 10)
    four = solve([10], [10], 4, [10 ** 6], [64], [4], 10)
    assert one[2] == [80] and one[1] == [3], f"80/64 -> 2 拍，prefill 1 拍，got {one}"
    assert four[2] == [140], f"80 + ceil(80*3/4)=60 -> 140，got {four[2]}"
    assert four[1] == [4], f"ceil(140/64)=3，prefill 1 拍 -> 4，got {four[1]}"
    big = solve([500], [500], 4, [10 ** 6], [64], [4], 10)
    assert big[2] == [7000], f"4000 + ceil(4000*3/4)=3000 -> 7000，got {big[2]}"


def check_link_serializes(solve):
    """同一副本上的两个请求排队：后一个的起点不早于前一个搬完。"""
    got = solve([10, 10], [5, 5], 4, [10 ** 6], [60], [1], 100)
    assert got[1] == [2, 3], f"各 60 字节 1 拍，第二个要等第一个搬完，got {got[1]}"
    assert got[2] == [120], f"got {got[2]}"


def check_prefill_cumulative_ceil(solve):
    """prefill 用累计 token 整体取整，不是每请求各自向上取整。"""
    got = solve([3, 3, 3], [0, 0, 0], 1, [10 ** 9], [10 ** 9], [1], 2)
    assert got[1] == [3, 4, 6], f"累计 3/6/9 -> 2/3/5 拍，再各加 1 拍搬运，got {got[1]}"
    per_request = [2, 4, 6]
    assert got[1] != per_request, "每请求各自取整会得到 2/4/6，与累计口径不同"


def check_rejected_still_consumes_prefill(solve):
    """被拒绝的请求不占显存，但它已经排过的 prefill 时间仍然计入后续请求。"""
    got = solve([10, 10, 10], [20, 0, 0], 4, [100], [40], [1], 1)
    assert got[0] == [-1, 0, 0], f"首个请求 120 字节超过容量 100，got {got[0]}"
    assert got[1] == [-1, 21, 31], f"第二个请求要等到第 20 拍 prefill 完，got {got[1]}"
    assert got[4] == 1 and got[5] == 31, f"got {got[4:]}"


def check_capacity_exact_fit(solve):
    """resident + kv == caps 算装得下；再多一个就不行。"""
    got = solve([10, 10, 10], [0, 0, 0], 4, [80], [1000], [1], 1000)
    assert got[0] == [0, 0, -1], f"40+40=80 刚好装满，第三个被拒，got {got[0]}"
    assert got[3] == [80] and got[4] == 1, f"got {got[3:]}"


def check_earliest_finish_wins(solve):
    """不是挑「最闲」的副本：副本 0 空闲更早，但链路慢，仍选搬运后更早完成的副本 1。"""
    got = solve([320, 100], [0, 0], 4, [10 ** 6, 10 ** 6], [8, 64], [1, 1], 1000)
    assert got[0] == [1, 1], f"副本 0 空闲在 0 拍但 400 字节要 50 拍，副本 1 只需 21+7，got {got[0]}"
    assert got[1] == [21, 28], f"got {got[1]}"


def check_identical_replicas(solve):
    """两个副本完全对称时，首个请求按下标小的优先；后续请求自然摊到空闲的那个。"""
    first = solve([10], [0], 4, [10 ** 6, 10 ** 6], [64, 64], [1, 1], 1000)
    assert first[0] == [0] and first[1] == [2], f"got {first}"
    got = solve([10, 10, 10], [0, 0, 0], 4, [10 ** 6, 10 ** 6], [64, 64], [1, 1], 1000)
    assert got[0] == [0, 1, 0], f"第三个请求回到更早空闲的副本 0，got {got[0]}"
    assert got[1] == [2, 2, 3], f"got {got[1]}"
    assert got[2] == [80, 40], f"got {got[2]}"


def check_all_rejected_and_empty(solve):
    """容量为 0 时全拒；空请求列表返回空账本。"""
    got = solve([5, 6], [0, 0], 4, [0], [8], [1], 10)
    assert got == ([-1, -1], [-1, -1], [0], [0], 2, 0), f"got {got}"
    empty = solve([], [], 4, [100], [64], [1], 10)
    assert empty == ([], [], [0], [0], 0, 0), f"got {empty}"


def check_bad_input(solve):
    """长度不一致、非法参数都要就地报错。"""
    for bad in (
        lambda: solve([1, 2], [0], 4, [10], [8], [1], 10),
        lambda: solve([1], [0], 4, [10, 10], [8], [1], 10),
        lambda: solve([0], [0], 4, [10], [8], [1], 10),
        lambda: solve([1], [-1], 4, [10], [8], [1], 10),
        lambda: solve([1], [0], 0, [10], [8], [1], 10),
        lambda: solve([1], [0], 4, [10], [0], [1], 10),
        lambda: solve([1], [0], 4, [10], [8], [0], 10),
        lambda: solve([1], [0], 4, [10], [8], [1], 0),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("应当抛出 ValueError")


def check_random_against_oracle(solve):
    """随机 60 组配置与独立 oracle 对拍。"""
    rng = random.Random(20261029)
    for _ in range(60):
        n = rng.randint(1, 8)
        d = rng.randint(1, 3)
        prompt = [rng.randint(1, 60) for _ in range(n)]
        gen = [rng.randint(0, 60) for _ in range(n)]
        kv = rng.randint(1, 8)
        caps = [rng.randint(0, 3000) for _ in range(d)]
        bws = [rng.randint(1, 128) for _ in range(d)]
        tps = [rng.randint(1, 8) for _ in range(d)]
        prefill_tps = rng.randint(1, 50)
        want = _oracle(prompt, gen, kv, caps, bws, tps, prefill_tps)
        got = solve(prompt, gen, kv, caps, bws, tps, prefill_tps)
        assert got == want, (
            f"不一致: prompt={prompt} gen={gen} kv={kv} caps={caps} bws={bws} "
            f"tps={tps} prefill_tps={prefill_tps}\n got={got}\nwant={want}"
        )
