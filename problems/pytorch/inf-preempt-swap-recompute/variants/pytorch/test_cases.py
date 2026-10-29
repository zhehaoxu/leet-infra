"""独立 oracle：按优先级分桶后逐层挑人，与参考实现的排序键写法不同。"""

import math
import random


def _oracle(kv, prompt, generated, priority, need, bytes_per_token, swap_bw, tick_per_token, freezable):
    buckets = {}
    for i, level in enumerate(priority):
        if freezable[i]:
            buckets.setdefault(level, []).append(i)

    victims = []
    modes = []
    freed = 0
    ticks = 0
    replay = 0
    for level in sorted(buckets):
        if freed >= need:
            break
        for i in sorted(buckets[level], key=lambda j: (kv[j], -j), reverse=True):
            if freed >= need:
                break
            swap_ticks = math.ceil(2 * kv[i] * bytes_per_token / swap_bw)
            recompute_ticks = (prompt[i] + generated[i]) * tick_per_token
            if recompute_ticks < swap_ticks:
                modes.append("recompute")
                ticks += recompute_ticks
                replay += prompt[i] + generated[i]
            else:
                modes.append("swap")
                ticks += swap_ticks
            victims.append(i)
            freed += kv[i]

    if freed < need:
        return [], [], freed, -1, 0
    return victims, modes, freed, ticks, replay


def check_example(solve):
    """一个牺牲者划得来换出（生成得多、重算贵），另一个划得来重算（几乎没生成）。"""
    got = solve(
        [4, 40, 40], [30, 5, 5], [10, 100, 1], [0, 0, 0], 44, 4, 16, 1, [True, True, True]
    )
    assert got[0] == [1, 2], f"kv 大的先挑，got {got[0]}"
    assert got[1] == ["swap", "recompute"], f"swap=ceil(2*40*4/16)=20，重算 105 与 6，got {got[1]}"
    assert got[2] == 80, f"got {got[2]}"
    assert got[3] == 26, f"20 + 6 = 26，got {got[3]}"
    assert got[4] == 6, f"只有一个重算牺牲者，重放 6 个 token，got {got[4]}"


check_example.example = True


def check_swap_cheaper(solve):
    """带宽充足、序列又长：换出（含换回）比重新 prefill 便宜。"""
    got = solve([40], [10], [8], [0], 40, 4, 64, 1, [True])
    assert got == ([0], ["swap"], 40, 5, 0), f"ceil(2*40*4/64)=5，got {got}"


def check_recompute_cheaper(solve):
    """带宽很窄、序列很短：直接丢掉 KV 重新 prefill 更划算。"""
    got = solve([40], [10], [0], [0], 40, 4, 4, 1, [True])
    assert got == ([0], ["recompute"], 40, 10, 10), f"换出要 80 拍，重算只要 10 拍，got {got}"


def check_swap_wins_tie(solve):
    """两者恰好相等时按约定选 swap。"""
    got = solve([8], [8], [0], [0], 8, 4, 8, 1, [True])
    assert got == ([0], ["swap"], 8, 8, 0), f"ceil(2*8*4/8)=8 与 8 打平，应选 swap，got {got}"


def check_tier_order(solve):
    """低优先级层先动，本层不够才升到下一层，不是全局按 kv 排序。"""
    got = solve([30, 6, 50], [10, 10, 10], [2, 2, 2], [2, 0, 1], 20, 4, 64, 1, [True, True, True])
    assert got[0] == [1, 2], f"优先级 0 的先动，再升到优先级 1，got {got[0]}"
    assert got[2] == 56, f"6 + 50 = 56，got {got[2]}"
    assert got[3] == 1 + 7, f"ceil(48/64)=1 与 ceil(400/64)=7，got {got[3]}"


def check_stops_when_enough(solve):
    """够用即停：不会为了「顺手多腾一点」去抢后续请求。"""
    got = solve([12, 18], [20, 4], [3, 0], [0, 0], 12, 4, 64, 1, [True, True])
    assert got[0] == [1], f"kv=18 的请求一枪就够，got {got[0]}"
    assert got[2] == 18, f"freed 允许大于 need，got {got[2]}"


def check_same_kv_smaller_index_first(solve):
    """同层同 kv 时按下标升序，保证结果唯一。"""
    got = solve([10, 10, 10], [5, 5, 5], [0, 0, 0], [0, 0, 0], 10, 4, 64, 1, [True, True, True])
    assert got[0] == [0], f"只抢一个就够，应选下标最小的，got {got[0]}"


def check_protected_skipped(solve):
    """受保护请求永不入选，哪怕它腾出的槽位更多。"""
    got = solve([30, 30, 10], [8, 8, 4], [2, 2, 1], [0, 1, 2], 20, 4, 64, 1, [True, False, False])
    assert got == ([0], ["swap"], 30, 4, 0), f"只有下标 0 可动，got {got}"


def check_infeasible(solve):
    """可抢占的槽位加起来都不够：整体放弃，ticks 记为 -1。"""
    got = solve([10, 20], [4, 4], [1, 1], [0, 1], 35, 4, 64, 1, [True, True])
    assert got == ([], [], 30, -1, 0), f"got {got}"


def check_random_against_oracle(solve):
    """随机实例与独立 oracle 全等，并核对账本不变量。"""
    rng = random.Random(4880)
    for _ in range(300):
        n = rng.randint(1, 6)
        kv = [rng.randint(1, 20) for _ in range(n)]
        prompt = [rng.randint(1, 16) for _ in range(n)]
        gen = [rng.randint(0, 12) for _ in range(n)]
        prio = [rng.randint(0, 3) for _ in range(n)]
        freezable = [rng.random() < 0.8 for _ in range(n)]
        need = rng.randint(1, sum(kv))
        bpt = rng.choice([1, 2, 4, 8])
        bw = rng.choice([1, 3, 16, 64])
        tpt = rng.choice([1, 2, 5])
        args = (kv, prompt, gen, prio, need, bpt, bw, tpt, freezable)

        got = solve(*args)
        want = _oracle(*args)
        assert got == want, f"{args}: {got} vs {want}"

        victims, modes, freed, ticks, replay = got
        assert len(victims) == len(modes), "victims 与 modes 必须等长"
        assert all(freezable[i] for i in victims), "不许抢占受保护请求"
        assert [prio[i] for i in victims] == sorted(prio[i] for i in victims), "应按优先级升序入选"
        if freed < need:
            assert (victims, modes, ticks, replay) == ([], [], -1, 0), f"不可行时须整体放弃：{got}"
        else:
            assert freed == sum(kv[i] for i in victims), f"腾出量与 victims 不符：{freed}"
            assert ticks >= replay * tpt, f"开销应不低于重放开销：{got}"
