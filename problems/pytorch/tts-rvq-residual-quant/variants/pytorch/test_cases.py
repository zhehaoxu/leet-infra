def _py(latents, codebooks):
    """oracle：独立实现（min+enumerate 取码、recon 先组装再求残差）。"""
    K = len(codebooks)
    D = len(latents[0])
    codes = []
    recon_all = []
    residual_sq = [[] for _ in range(K)]
    for v in latents:
        residual = list(map(float, v))
        picks = []
        for k in range(K):
            cb = codebooks[k]
            dists = []
            for entry in cb:
                d = sum((residual[j] - entry[j]) ** 2 for j in range(D))
                dists.append(d)
            idx = dists.index(min(dists))  # 并列取首个（最小下标）
            picks.append(idx)
            for j in range(D):
                residual[j] -= cb[idx][j]
            for lvl in range(k, K):
                pass
            residual_sq[k].append(sum(x * x for x in residual))
        recon = [0.0] * D
        for k in range(K):
            for j in range(D):
                recon[j] += codebooks[k][picks[k]][j]
        recon_all.append(recon)
        codes.append(picks)  # [N][K]
    # 转成 [K][N]
    codes_kn = [[codes[n][k] for n in range(len(latents))] for k in range(K)]
    return {"codes": codes_kn, "reconstructed": recon_all,
            "residual_sq": residual_sq}


def _compare(solve, latents, codebooks):
    led = solve(latents, codebooks)
    ref = _py(latents, codebooks)
    # codes 严格；浮点容差
    assert led["codes"] == ref["codes"], f"codes:\n{led['codes']}\nvs\n{ref['codes']}"
    for a, b in zip(led["reconstructed"], ref["reconstructed"]):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert abs(x - y) < 1e-9, f"recon {x} vs {y}"
    for ra, rb in zip(led["residual_sq"], ref["residual_sq"]):
        for x, y in zip(ra, rb):
            assert abs(x - y) < 1e-9, f"resid {x} vs {y}"
    return led


def check_anchor_hand(solve):
    # 1-D 两级码本手算：latent=1.05
    # 级0 码本 [0.0, 1.0, 2.0]：距 1.0 最近（|0.05| vs |1.05|/|0.95|）→ idx1，残差 0.05
    # 级1 码本 [0.05, -0.05]：→ idx0，残差 0
    led = _compare(solve, [[1.05]], [[[0.0], [1.0], [2.0]], [[0.05], [-0.05]]])
    assert led["codes"] == [[1], [0]]
    assert led["reconstructed"] == [[1.05]]
    assert led["residual_sq"][0] == [0.0025000000000000044] or abs(led["residual_sq"][0][0] - 0.0025) < 1e-12
    assert abs(led["residual_sq"][1][0]) < 1e-30  # 浮点残差 ~1.7e-33


def check_residual_cascade(solve):
    # 级1 修正级0 的残差方向
    led = _compare(solve, [[0.3]], [[[0.0], [1.0]], [[0.3], [-0.3]]])
    # 级0 最近 0.0（|0.3| < |0.7|）→ 残差 0.3；级1 → 0.3 → 残差 0
    assert led["codes"] == [[0], [0]]
    assert abs(led["reconstructed"][0][0] - 0.3) < 1e-12
    assert abs(led["residual_sq"][1][0]) < 1e-30  # 浮点残差 ~1.7e-33


def check_tie_low_index(solve):
    # 等距并列：0.5 到 0.0 与 1.0 等距 → 取下标 0
    led = _compare(solve, [[0.5]], [[[0.0], [1.0]]])
    assert led["codes"] == [[0]]


def check_multi_sample_independent(solve):
    led = _compare(solve, [[1.05], [-2.4]],
                   [[[0.0], [1.0], [2.0], [-2.0]], [[0.05], [-0.05], [0.4]]])
    assert led["codes"][0] == [1, 3]      # 样本0→1.0、样本1→-2.0
    assert led["codes"][1] == [0, 1]      # 残差 0.05→0.05(idx0)；残差 -0.4 最近 0.4? 距 0.45/0.35/0.8 → idx1(-0.05)
    assert abs(led["residual_sq"][1][0]) < 1e-30
    assert abs(led["residual_sq"][1][1] - 0.1225) < 1e-9


def check_multidim(solve):
    # 2-D 向量：码本条目正交方向
    led = _compare(solve, [[0.9, 0.2]], [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.1], [0.1, 0.0]]])
    # 级0 最近 [1,0] → 残差 [-0.1, 0.2]；级1 距 [0,0.1]: 0.01+0.01=0.02、距 [0.1,0]: 0.04+0.04=0.08 → idx0
    assert led["codes"] == [[0], [0]]
    assert abs(led["reconstructed"][0][0] - 1.0) < 1e-12
    assert abs(led["reconstructed"][0][1] - 0.1) < 1e-12
    assert abs(led["residual_sq"][1][0] - 0.02) < 1e-12  # 残差 [-0.1, 0.1]


def check_error_monotone_decrease(solve):
    # 残差能量逐级不增（每级吸收一部分）
    led = _compare(solve, [[0.6, 0.6]],
                   [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    s0, s1 = led["residual_sq"][0][0], led["residual_sq"][1][0]
    assert s1 < s0
    assert abs(led["reconstructed"][0][0] - 1.0) < 1e-12
    assert abs(led["reconstructed"][0][1] - 1.0) < 1e-12


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], [[[0.0]]])
    ve([[1.0]], [])
    ve([[]], [[[0.0]]])
    ve([[1.0], [1.0, 2.0]], [[[0.0, 0.0]]])   # 维度不齐
    ve([[1.0, 2.0]], [[[0.0]]])                # 码本条目维度不符
    ve([["x"]], [[[0.0]]])
    ve([[1.0]], [[[]]])                        # 空码本
    ve([[1.0]], [[[0.0]], []])                 # 某级空
    ve([[float("nan")]], [[[0.0]]])
    ve([[1.0]], [[[True, 0.0]]])
    ve("nope", [[[0.0]]])


def check_oracle_random(solve):
    import random
    rng = random.Random(20260673)
    for _ in range(150):
        N = rng.randint(1, 5)
        D = rng.randint(1, 3)
        K = rng.randint(1, 4)
        C = rng.randint(1, 5)
        latents = [[round(rng.uniform(-2, 2), 3) for _ in range(D)] for _ in range(N)]
        codebooks = [[[round(rng.uniform(-2, 2), 3) for _ in range(D)]
                      for _ in range(C)] for _ in range(K)]
        led = solve(latents, codebooks)
        ref = _py(latents, codebooks)
        assert led["codes"] == ref["codes"], "random codes mismatch"
        for a, b in zip(led["reconstructed"], ref["reconstructed"]):
            for x, y in zip(a, b):
                assert abs(x - y) < 1e-9
        for ra, rb in zip(led["residual_sq"], ref["residual_sq"]):
            for x, y in zip(ra, rb):
                assert abs(x - y) < 1e-9
