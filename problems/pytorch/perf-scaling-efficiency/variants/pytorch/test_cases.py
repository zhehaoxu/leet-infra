import torch


def _close(a, b, tol=1e-5):
    b = float(b)
    return abs(float(a) - b) <= tol * max(1.0, abs(b))


# 独立重算（对拍口径）：参考解从「整机吞吐」定义出发做两次除法；这里改走闭式合并
#   strong: speedup = t0/t_i, efficiency = (t0*w0)/(t_i*w_i)   （一次除法）
#   weak  : efficiency = t0/t_i, speedup = (w_i*t0)/(w_0*t_i)
# 超线性与饱和点则完全避开浮点除法，用整数/浮点的交叉相乘比较：
#   strong: eff>1 <=> t0*w0 > t_i*w_i ; 饱和 <=> t_i >= t_{i-1}
#   weak  : eff>1 <=> t0 > t_i        ; 饱和 <=> w_i*t_{i-1} <= w_{i-1}*t_i
def _oracle(world_sizes, step_times_ms, mode):
    w = [int(x) for x in world_sizes]
    t = [float(x) for x in step_times_ms]
    n = len(w)
    strong = (mode == "strong")
    speedup, eff = [], []
    for i in range(n):
        if strong:
            speedup.append(t[0] / t[i])
            eff.append((t[0] * w[0]) / (t[i] * w[i]))
        else:
            speedup.append((w[i] * t[0]) / (w[0] * t[i]))
            eff.append(t[0] / t[i])
    super_idx = []
    for i in range(n):
        hit = (t[0] * w[0] > t[i] * w[i]) if strong else (t[0] > t[i])
        if hit:
            super_idx.append(i)
    saturation = -1
    for i in range(1, n):
        hit = (t[i] >= t[i - 1]) if strong else (w[i] * t[i - 1] <= w[i - 1] * t[i])
        if hit:
            saturation = i
            break
    return (torch.tensor(speedup, dtype=torch.float64),
            torch.tensor(eff, dtype=torch.float64),
            torch.tensor(super_idx, dtype=torch.int64),
            int(saturation))


def _assert_tuple(got, want):
    assert len(got) == 4, got
    sp, eff, sup, sat = got
    assert sp.shape == want[0].shape and eff.shape == want[1].shape
    assert torch.allclose(sp, want[0], atol=1e-5, rtol=1e-5), (sp.tolist(), want[0].tolist())
    assert torch.allclose(eff, want[1], atol=1e-5, rtol=1e-5), (eff.tolist(), want[1].tolist())
    assert torch.equal(sup, want[2]), (sup.tolist(), want[2].tolist())
    assert int(sat) == int(want[3]), (sat, want[3])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    sp, eff, sup, sat = solve([1, 2, 4, 8, 16], [100.0, 45.0, 25.0, 15.0, 9.0], "strong")
    assert torch.allclose(sp, torch.tensor([1.0, 100 / 45, 4.0, 100 / 15, 100 / 9], dtype=torch.float64),
                          atol=1e-5, rtol=1e-5), sp.tolist()
    assert torch.allclose(eff, torch.tensor([1.0, 100 / 90, 1.0, 100 / 120, 100 / 144], dtype=torch.float64),
                          atol=1e-5, rtol=1e-5), eff.tolist()
    assert sup.tolist() == [1], sup.tolist()      # 只有 2 卡那一档效率 >1（超线性）
    assert int(sat) == -1, sat                    # 吞吐一路上升，没饱和
    sp, eff, sup, sat = solve([1, 2, 4, 8], [100.0, 105.0, 115.0, 240.0], "weak")
    assert torch.allclose(eff, torch.tensor([1.0, 100 / 105, 100 / 115, 100 / 240], dtype=torch.float64),
                          atol=1e-5, rtol=1e-5), eff.tolist()
    assert torch.allclose(sp, eff * torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64),
                          atol=1e-5, rtol=1e-5), sp.tolist()
    assert sup.tolist() == [], sup.tolist()
    assert int(sat) == 3, sat                     # 8 卡整机吞吐反而低于 4 卡 -> 饱和点


def check_matches_oracle(solve):
    cases = [
        ([1, 2, 4, 8, 16], [100.0, 45.0, 25.0, 15.0, 9.0], "strong"),
        ([1, 2, 4, 8], [100.0, 105.0, 115.0, 240.0], "weak"),
        ([1, 2, 4, 8], [100.0, 50.0, 25.0, 12.5], "strong"),
        ([1, 2, 4, 8], [100.0, 100.0, 100.0, 100.0], "weak"),
        ([8], [42.0], "strong"),
        ([1, 2, 4, 8], [100.0, 50.0, 25.0, 25.0], "strong"),
        ([2, 8, 32], [12.5, 3.5, 1.2], "weak"),
        ([4, 8, 16, 64, 128, 256], [80.0, 42.0, 25.0, 14.0, 13.0, 30.0], "strong"),
        ([1, 4], [1000.0, 260.0], "strong"),
        ([1, 2, 4, 8, 16, 32], [200.0, 210.0, 215.0, 230.0, 228.0, 240.0], "weak"),
    ]
    for c in cases:
        _assert_tuple(solve(*c), _oracle(*c))


def check_perfect_strong_scaling(solve):
    # 完美强扩展：卡数翻倍耗时减半 -> speedup = w/w0，效率恒 1（严格大于才算超线性）
    sp, eff, sup, sat = solve([1, 2, 4, 8], [100.0, 50.0, 25.0, 12.5], "strong")
    assert torch.allclose(sp, torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)), sp.tolist()
    assert torch.allclose(eff, torch.ones(4, dtype=torch.float64)), eff.tolist()
    assert sup.numel() == 0, sup.tolist()
    assert int(sat) == -1, sat


def check_perfect_weak_scaling(solve):
    # 完美弱扩展：每卡工作量固定 -> 耗时不变、效率恒 1、吞吐线性上升
    sp, eff, sup, sat = solve([1, 2, 4, 8], [100.0, 100.0, 100.0, 100.0], "weak")
    assert torch.allclose(sp, torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)), sp.tolist()
    assert torch.allclose(eff, torch.ones(4, dtype=torch.float64)), eff.tolist()
    assert sup.numel() == 0 and int(sat) == -1, (sup.tolist(), sat)


def check_plateau_is_saturation(solve):
    # 8 卡与 4 卡耗时相同：吞吐不再上升，饱和点落在第一处「加卡无收益」的档位
    sp, eff, sup, sat = solve([1, 2, 4, 8], [100.0, 50.0, 25.0, 25.0], "strong")
    assert torch.allclose(sp, torch.tensor([1.0, 2.0, 4.0, 4.0], dtype=torch.float64)), sp.tolist()
    assert torch.allclose(eff, torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=torch.float64)), eff.tolist()
    assert int(sat) == 3, sat
    assert sup.numel() == 0, sup.tolist()


def check_single_point_degenerate(solve):
    for mode in ("strong", "weak"):
        sp, eff, sup, sat = solve([8], [42.0], mode)
        assert sp.tolist() == [1.0] and eff.tolist() == [1.0], (mode, sp, eff)
        assert sup.numel() == 0 and int(sat) == -1, (mode, sup, sat)


def check_speedup_efficiency_identity(solve):
    # 两种模式下都恒有 speedup[i] == efficiency[i] * w[i]/w[0]
    for w, t, mode in [([1, 2, 4, 8, 16], [100.0, 45.0, 25.0, 15.0, 9.0], "strong"),
                       ([1, 2, 4, 8], [100.0, 105.0, 115.0, 240.0], "weak"),
                       ([3, 6, 12, 24], [12.5, 6.5, 3.5, 3.6], "strong"),
                       ([2, 8, 32], [12.5, 3.5, 1.2], "weak")]:
        sp, eff, _, _ = solve(w, t, mode)
        ratio = torch.tensor([float(x) / float(w[0]) for x in w], dtype=torch.float64)
        assert torch.allclose(sp, eff * ratio, atol=1e-5, rtol=1e-5), (mode, sp.tolist(), eff.tolist())
        assert eff[0].item() == 1.0 and sp[0].item() == 1.0, (mode, sp, eff)


def check_superlinear_matches_efficiency(solve):
    # superlinear 必须恰好是 efficiency > 1 的下标集合（升序、严格）
    for w, t, mode in [([1, 2, 4, 8, 16], [100.0, 45.0, 25.0, 15.0, 9.0], "strong"),
                       ([1, 2, 4, 8], [100.0, 90.0, 47.0, 22.0], "strong"),
                       ([1, 2, 4, 8], [100.0, 99.0, 96.0, 90.0], "weak"),
                       ([1, 2, 4], [10.0, 10.0, 10.0], "weak")]:
        sp, eff, sup, sat = solve(w, t, mode)
        expect = [i for i in range(len(w)) if float(eff[i]) > 1.0]
        assert sup.tolist() == expect, (mode, w, t, eff.tolist(), sup.tolist())
        for i in sup.tolist():
            assert float(eff[i]) > 1.0 + 1e-12, (i, eff.tolist())
        assert sp.numel() == eff.numel() == len(w), (sp, eff)
        assert int(sat) in range(-1, len(w)), sat


def check_mode_relation(solve):
    # 同一组数据：strong 的效率 = weak 的效率 * w0/w（weak 只按耗时比、strong 再摊卡数）
    w, t = [1, 2, 4, 8], [100.0, 105.0, 115.0, 240.0]
    _, eff_s, _, _ = solve(w, t, "strong")
    _, eff_w, _, _ = solve(w, t, "weak")
    for i in range(len(w)):
        expect = float(eff_w[i]) * float(w[0]) / float(w[i])
        assert _close(eff_s[i], expect), (i, eff_s.tolist(), eff_w.tolist())


check_example.example = True
