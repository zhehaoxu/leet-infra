"""SPP-Net 金字塔池化账（test cases）。

独立 oracle：逐 bin 显式枚举窗口（含重复窗口的多次贡献），与 solution
的循环重放互证；另加论文式 floor/ceil 对照（两者可不同）。
"""

import math
import random


def _py_solve(feature, bin_counts):
    h, w = len(feature), len(feature[0])
    pooled = []
    shapes, kernels, pads = [], [], []
    for k in bin_counts:
        kh = -(-h // k)          # 整数 ceil
        kw = -(-w // k)
        padh = (kh * k - h + 1) // 2
        padw = (kw * k - w + 1) // 2
        kernels.append([kh, kw])
        pads.append([padh, padw])
        shapes.append([k, k])
        for ph in range(k):
            for pw in range(k):
                hs = max(0, ph * kh - padh)
                ws = max(0, pw * kw - padw)
                he = min(hs + kh, h)
                we = min(ws + kw, w)
                pooled.append(max(
                    (feature[r][c]
                     for r in range(hs, he)
                     for c in range(ws, we)),
                    default=-math.inf,     # 官方空 bin 输出 −FLT_MAX
                ))
    return {"pooled": pooled, "shapes": shapes, "kernels": kernels, "pads": pads}


def _paper_windows(h, k):
    """SPP 论文式：hs=floor(i·H/k), he=ceil((i+1)·H/k)（非重叠）。"""
    out = []
    for i in range(k):
        out.append((math.floor(i * h / k), math.ceil((i + 1) * h / k)))
    return out


def _close(actual, expect, tol=1e-9):
    if isinstance(expect, dict):
        assert isinstance(actual, dict) and len(actual) == len(expect)
        return all(_close(actual[k], expect[k], tol) for k in expect)
    if isinstance(expect, list):
        assert isinstance(actual, list) and len(actual) == len(expect), \
            f"{actual!r} vs {expect!r}"
        return all(_close(a, e, tol) for a, e in zip(actual, expect))
    if isinstance(expect, float):
        if math.isinf(expect):
            assert math.isinf(actual) and (actual > 0) == (expect > 0), \
                f"{actual} != {expect}"
            return True
        assert abs(actual - expect) < tol, f"{actual} != {expect}"
        return True
    assert actual == expect, f"{actual!r} != {expect!r}"
    return True


def _compare(solve, feature, bin_counts):
    led = solve(feature, bin_counts)
    _close(led, _py_solve(feature, bin_counts))
    return led


def check_paper_example_5x5(solve):
    # 论文经典 5×5 + [1,2,4] → 1+4+16=21 维
    f = [[float(5 * r + c) for c in range(5)] for r in range(5)]
    led = _compare(solve, f, [1, 2, 4])
    assert led["shapes"] == [[1, 1], [2, 2], [4, 4]]
    assert len(led["pooled"]) == 21
    # 全局 level：max=24；k=1 kernel=5,pad=0
    assert led["pooled"][0] == 24.0
    _close(led["kernels"][0], [5, 5])
    _close(led["pads"][0], [0, 0])
    # k=2：kernel=ceil(5/2)=3, pad=(6−5+1)//2=1
    _close(led["kernels"][1], [3, 3])
    _close(led["pads"][1], [1, 1])
    # k=2 bin(0,0)：hs=max(0,−1)=0, he=min(3,5)=3 → max(0..2,0..2)=12
    assert led["pooled"][1] == 12.0
    # k=4：kernel=2, pad=(8−5+1)//2=2；bin(0,0) hs=max(0,−2)=0, he=2 → 6
    _close(led["kernels"][2], [2, 2])
    _close(led["pads"][2], [2, 2])
    assert led["pooled"][5] == 6.0


def check_rectangular_wh(solve):
    # H=4, W=7：kernel 分轴算（k=3 → ceil(4/3)=2, ceil(7/3)=3）
    f = [[float(r * 7 + c) for c in range(7)] for r in range(4)]
    led = _compare(solve, f, [3])
    _close(led["kernels"][0], [2, 3])
    _close(led["pads"][0], [1, 1])
    # bin(0,0)：hs=max(0,−1)=0, he=min(3,4)=2；ws=max(0,−1)=0, we=min(3,7)=3
    # → 行 0..1 列 0..2 → max=9
    assert led["pooled"][0] == 9.0
    # bin(2,2) h 侧：hs=max(0,4−1)=3, he=min(5,4)=4 → 行 3
    #               w 侧：ws=max(0,6−1)=5, we=min(8,7)=7 → 列 5,6 → max 26,27
    assert led["pooled"][8] == 27.0


def check_kernel_equals_h(solve):
    # k > H/2 时窗口重叠（H=3, k=2）：kernel=2, pad=(4−3+1)//2=1
    f = [[1.0, 9.0, 3.0], [2.0, 8.0, 4.0], [7.0, 6.0, 5.0]]
    led = _compare(solve, f, [2])
    _close(led["pads"][0], [1, 1])
    # bin(0,0): hs=0, he=2 → max(1,9,2,8)=9；bin(0,1): 列 1,2 → max(9,3,8,4)=9
    assert led["pooled"][0] == 9.0
    assert led["pooled"][1] == 9.0
    # bin(1,0): hs=max(0,2−1)=1, he=3 → 行 1,2 列 0,1 → max(2,8,7,6)=8
    assert led["pooled"][2] == 8.0


def check_clip_bounds(solve):
    # pad 使 bin(0,·) 窗口起点为负（clip 0）、末 bin 终点 clip H
    f = [[float(10 * r + c) for c in range(6)] for r in range(6)]
    led = _compare(solve, f, [4])
    _close(led["kernels"][0], [2, 2])
    _close(led["pads"][0], [1, 1])
    # bin(0,0)：hs=ws=max(0,−1)=0 → max(0,1,10,11)=11
    assert led["pooled"][0] == 11.0
    # bin(3,3)：hs=ws=max(0,6−1)=5, he=we=min(7,6)=6 → 55
    assert led["pooled"][15] == 55.0


def check_differs_from_paper_formula(solve):
    # H=5, k=4：caffe pad 式与论文 floor/ceil 式的窗口不同
    # caffe：kernel=2,pad=2 → hs=[0,0,2,4], he=[2,2,4,5]
    # 论文：hs=[0,1,2,3], he=[2,2,4,5]
    f = [[float(5 * r + c) for c in range(5)] for r in range(5)]
    led = _compare(solve, f, [4])
    win = _paper_windows(5, 4)
    assert win[0] == (0, 2) and win[1] == (1, 3)   # 论文式 bin1 从 1 起
    # caffe 式 bin(1,0)：hs=max(0,2−2)=0 → 与 bin(0,0) 同窗 → max 相同
    assert led["pooled"][5] == led["pooled"][4] == 6.0


def check_empty_bin_official(solve):
    # H=6, k=5：官方 ceil-kernel+pad 式在末 bin 产生空窗口（hs=6≥H）
    # → MaxPoolForward 输出 −FLT_MAX（此处 −inf）
    f = [[float(9 * r + c) for c in range(6)] for r in range(6)]
    led = _compare(solve, f, [5])
    _close(led["kernels"][0], [2, 2])
    _close(led["pads"][0], [2, 2])
    # bin(4,·)：hs=4·2−2=6 ≥ H → 空 bin
    assert led["pooled"][20] == -math.inf
    assert led["pooled"][24] == -math.inf
    # 非空 bin 正常
    assert led["pooled"][0] == 10.0


def check_single_level_and_value(solve):
    led = _compare(solve, [[3.5]], [1])
    assert led["pooled"] == [3.5]
    assert led["kernels"] == [[1, 1]] and led["pads"] == [[0, 0]]


def check_oracle_random(solve):
    rng = random.Random(703)
    for _ in range(200):
        h = rng.randint(1, 12)
        w = rng.randint(1, 12)
        kmax = min(h, w)
        counts = sorted(rng.sample(range(1, kmax + 1),
                                   rng.randint(1, min(3, kmax))))
        f = [[round(rng.uniform(-10, 10), 3) for _ in range(w)]
             for _ in range(h)]
        _compare(solve, f, counts)


def check_invalid_inputs(solve):
    for args in [
        ([], [1]),
        ([[1.0, 2.0], [3.0]], [1]),               # 行宽不齐
        ([[1.0, 2.0]], [3]),                      # H < max bin
        ([[1.0], [2.0]], [3]),                    # W < max bin
        ([[1.0, 2.0], [3.0, 4.0]], []),
        ([[1.0, 2.0], [3.0, 4.0]], [2, 0]),       # bin 0
        ([[1.0, 2.0], [3.0, 4.0]], [1.5]),
        ([[1.0, float("nan")], [3.0, 4.0]], [1]),
        ([[1.0], []], [1]),                       # 空行
    ]:
        try:
            solve(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"expected error: {args}")
