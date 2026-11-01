"""order 694: FunASR `cif()` fire 发射账。

oracle 用解析法独立实现：帧贡献区间 [c_{t-1}, c_t) 与 token 区间
[k-1, k) 的相交长度作权重——不经状态机。随机对拍用 m/256 网格
（2 的幂分母，浮点精确）避免边界翻转。
"""
import math
import random


def _oracle(alphas, hidden, threshold=1.0, tail_threshold=0.45, use_tail=True):
    if use_tail:
        alphas = list(alphas) + [tail_threshold]
        hidden = [list(h) for h in hidden] + [[0.0] * len(hidden[0])]

    T = len(alphas)
    D = len(hidden[0])
    # 帧贡献区间（半开）：[c[t-1], c[t])
    c = [0.0] * (T + 1)
    for t in range(T):
        c[t + 1] = c[t] + alphas[t]

    n_fire = int(math.floor(c[T] + 1e-9))  # 发射数 = 跨越的整数格点数
    embeds = []
    fire_positions = []
    for k in range(1, n_fire + 1):
        lo, hi = k - 1.0, float(k)
        emb = [0.0] * D
        fire_t = None
        for t in range(T):
            w = min(c[t + 1], hi) - max(c[t], lo)
            if w > 0:
                for d in range(D):
                    emb[d] += w * hidden[t][d]
            if fire_t is None and c[t + 1] >= hi - 1e-9:
                fire_t = t  # 首个使累积达到 k 的帧
        embeds.append(emb)
        fire_positions.append(fire_t)

    total = sum(alphas)
    token_num = int(math.floor(total + 1e-9)) if use_tail else \
        int(math.floor(total + 0.5 + 1e-9))
    return {"embeds": embeds, "fire_positions": fire_positions,
            "token_num": token_num}


def _compare(solve, alphas, hidden, threshold=1.0, tail_threshold=0.45,
             use_tail=True):
    led = solve(list(alphas), [list(h) for h in hidden], threshold,
                tail_threshold, use_tail)
    exp = _oracle(list(alphas), [list(h) for h in hidden], threshold,
                  tail_threshold, use_tail)
    assert led["fire_positions"] == exp["fire_positions"], \
        f"{led}\n{exp}\nalphas={alphas}"
    assert led["token_num"] == exp["token_num"], f"{led}\n{exp}"
    assert len(led["embeds"]) == len(exp["embeds"]), f"{led}\n{exp}"
    for a, b in zip(led["embeds"], exp["embeds"]):
        assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
                   for x, y in zip(a, b)), f"{led}\n{exp}"
    return led


H1 = [[1.0, 0.0]]
H3 = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
H4 = H3 + [[3.0, 3.0]]


def check_uniform_halves(solve):
    led = _compare(solve, [0.5, 0.5, 0.5, 0.5], H4, use_tail=False)
    assert led["fire_positions"] == [1, 3]
    assert all(math.isclose(a, b) for a, b in
               zip(led["embeds"][0], [0.5, 0.5]))
    assert all(math.isclose(a, b) for a, b in
               zip(led["embeds"][1], [2.5, 2.5]))
    assert led["token_num"] == 2  # round(2.0)


def check_remainder_dropped_no_tail(solve):
    # [0.6,0.6,0.6] 无 tail：fire 仅 t1，embed=0.6h0+0.4h1，余 0.2 丢弃
    led = _compare(solve, [0.6, 0.6, 0.6], H3, use_tail=False)
    assert led["fire_positions"] == [1]
    assert all(math.isclose(a, b) for a, b in zip(led["embeds"][0], [0.6, 0.4]))
    # 口径差：发射数 1 ≠ round(1.8)=2
    assert len(led["embeds"]) == 1 and led["token_num"] == 2


def check_tail_fires_final_token(solve):
    # [0.6,0.6,0.6] + tail 0.45：fire t1、t3(0.8+0.45=1.25)，tail 帧对零行
    led = _compare(solve, [0.6, 0.6, 0.6], H3, use_tail=True)
    assert led["fire_positions"] == [1, 3]
    assert led["token_num"] == 2  # floor(2.25)
    # 末 token = 0.2h1 + 0.6h2（tail 缺口 0.2 乘零行不贡献）
    assert all(math.isclose(a, b) for a, b in zip(led["embeds"][1], [1.2, 1.4]))


def check_exact_boundary_fires(solve):
    # 累积恰好 1.0 也 fire（闭区间）：[0.2]*5 → 仅 t4
    H5 = H4 + [[4.0, 4.0]]
    led = _compare(solve, [0.2] * 5, H5, use_tail=False)
    assert led["fire_positions"] == [4]
    assert all(math.isclose(a, b) for a, b in zip(led["embeds"][0], [2.0, 2.0]))
    # 单帧 alpha=1.0：缺口为 1，整帧发射
    led2 = _compare(solve, [1.0], H1, use_tail=False)
    assert led2["fire_positions"] == [0]
    assert all(math.isclose(a, b) for a, b in zip(led2["embeds"][0], [1.0, 0.0]))
    # 三帧各 1.0 连发，每帧独立成 token
    led3 = _compare(solve, [1.0, 1.0, 1.0], H3, use_tail=False)
    assert led3["fire_positions"] == [0, 1, 2]
    assert all(math.isclose(a, b) for a, b in
               zip(led3["embeds"][2], [2.0, 2.0]))


def check_tail_exact_integer_boundary(solve):
    # 尾余 0.55 + tail 0.45 = 1.0 → tail 帧（零向量行）fire
    H1b = [[1.0, 2.0]]
    led = _compare(solve, [0.55], H1b, use_tail=True)
    assert led["fire_positions"] == [1]
    assert led["token_num"] == 1  # floor(1.0)
    # embed = 0.55*h0 + 0.45*零行
    assert all(math.isclose(a, b) for a, b in zip(led["embeds"][0], [0.55, 1.1]))


def check_all_zero_no_fire(solve):
    led = _compare(solve, [0.0, 0.0, 0.0], H3, use_tail=False)
    assert led["embeds"] == [] and led["fire_positions"] == []
    led2 = _compare(solve, [0.0, 0.0, 0.0], H4, use_tail=True)
    # +0.45 仍不足 1：无发射；token_num = floor(0.45) = 0
    assert led2["embeds"] == [] and led2["token_num"] == 0


def check_fire_snapshot_before_decrement(solve):
    # 快照在减 1 前：跨阈帧 integrate 快照 1.8 ≥ 1 即 fire；
    # 若错用减后值 0.8/0.7 判定则发射丢失——由随机对拍兜底，这里验证多连发
    H6 = H4 + [[4.0, 4.0], [5.0, 5.0]]
    led = _compare(solve, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], H6, use_tail=False)
    # integrate: 0.9, 1.8(f), 1.7(f), 1.6(f), 1.5(f), 1.4(f) —— 0.9 连续帧
    # 每帧都跨阈（fire 后余 0.8+0.9=1.7 ≥ 1），fire at 1..5
    assert led["fire_positions"] == [1, 2, 3, 4, 5]
    assert led["token_num"] == 5  # round(5.4)


def check_random_grid_against_oracle(solve):
    # m/256 网格：2 的幂分母，浮点精确，边界判定无翻转
    rng = random.Random(694)
    D = 3
    for _ in range(300):
        T = rng.randint(1, 10)
        alphas = [rng.randint(0, 255) / 256 for _ in range(T)]
        hidden = [[float(rng.randint(-5, 5)) for _ in range(D)]
                  for _ in range(T)]
        tt = rng.randint(0, 255) / 256
        use_tail = rng.random() < 0.5
        led = solve(list(alphas), [list(h) for h in hidden],
                    1.0, tt, use_tail)
        exp = _oracle(list(alphas), [list(h) for h in hidden], 1.0, tt, use_tail)
        assert led["fire_positions"] == exp["fire_positions"], \
            f"{led}\n{exp}\n{alphas}"
        assert led["token_num"] == exp["token_num"]
        assert len(led["embeds"]) == len(exp["embeds"])
        for a, b in zip(led["embeds"], exp["embeds"]):
            assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
                       for x, y in zip(a, b)), f"{led}\n{exp}"
        # 单序列恒等式：use_tail 时发射数 = token_num
        if use_tail:
            assert len(led["embeds"]) == led["token_num"]
        # fire_positions 升序且严格递增
        assert all(p < q for p, q in
                   zip(led["fire_positions"], led["fire_positions"][1:]))
