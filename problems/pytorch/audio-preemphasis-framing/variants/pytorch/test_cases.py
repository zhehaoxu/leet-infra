import math
import random

import torch


def _ref(wave, preemph, frame_len, hop_len, window):
    """独立 oracle：逐点递推预加重 + 逐帧 Python 循环，不用任何张量切片技巧。"""
    x = [float(v) for v in wave]
    y = [x[0]]
    for t in range(1, len(x)):
        y.append(x[t] - preemph * x[t - 1])

    n = frame_len
    if n == 1:
        win = [1.0]
    elif window == "hann":
        win = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]
    elif window == "hamming":
        win = [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]
    else:
        win = [1.0] * n

    n_frames = 1 + (len(x) - frame_len) // hop_len
    frames = [
        [y[f * hop_len + i] * win[i] for i in range(n)] for f in range(n_frames)
    ]
    return y, frames


def check_example(solve):
    torch.manual_seed(0)
    wave = torch.tensor([0.5, -0.25, 0.75, 0.0, -1.0, 0.25, 1.0, 0.5, -0.5, 0.125])
    y, frames = solve(wave, 0.97, 4, 3, "hamming")

    want_y = torch.tensor(
        [0.5, -0.735, 0.9925, -0.7275, -1.0, 1.22, 0.7575, -0.47, -0.985, 0.61]
    )
    assert y.shape == (10,), f"y 形状应为 (10,)，got {tuple(y.shape)}"
    assert torch.allclose(y, want_y, atol=1e-6), f"预加重结果不对：{y.tolist()}"

    want_frames = torch.tensor(
        [
            [0.04, -0.56595, 0.764225, -0.0582],
            [-0.0582, -0.77, 0.9394, 0.0606],
            [0.0606, -0.3619, -0.75845, 0.0488],
        ]
    )
    assert frames.shape == (3, 4), f"帧矩阵形状应为 (3, 4)，got {tuple(frames.shape)}"
    assert torch.allclose(frames, want_frames, atol=1e-6), f"加窗帧不对：{frames.tolist()}"


check_example.example = True


def check_hann_edges(solve):
    """对称 Hann 窗两端为 0：无论信号多大，每帧首尾都被压成 0。"""
    torch.manual_seed(1)
    wave = torch.randn(20)
    _, frames = solve(wave, 0.9, 8, 4, "hann")
    assert frames.shape == (4, 8), f"帧数应为 1+(20-8)//4=4，got {tuple(frames.shape)}"
    assert torch.all(frames[:, 0].abs() < 1e-7), "hann 窗首点为 0"
    assert torch.all(frames[:, -1].abs() < 1e-7), "hann 窗末点为 0"

    # Hamming 窗两端都是 0.08（0.54 - 0.46）
    y, ham = solve(wave, 0.9, 8, 3, "hamming")
    starts = torch.arange(ham.shape[0]) * 3
    assert torch.allclose(ham[:, 0], 0.08 * y[starts], atol=1e-6), "hamming 首点权重 0.08"
    assert torch.allclose(ham[:, -1], 0.08 * y[starts + 7], atol=1e-6), "hamming 末点权重 0.08"


def check_rect_and_no_preemphasis(solve):
    """window="rect" 且 preemph=0 时，帧就是波形的连续切片。"""
    torch.manual_seed(2)
    wave = torch.randn(15)
    y, frames = solve(wave, 0.0, 5, 2, "rect")
    assert torch.allclose(y, wave, atol=1e-7), "preemph=0 时预加重不应改变波形"
    for f in range(frames.shape[0]):
        seg = wave[f * 2 : f * 2 + 5]
        assert torch.allclose(frames[f], seg, atol=1e-7), f"第 {f} 帧不是原样切片"


def check_random_vs_oracle(solve):
    """随机负载对拍逐点递推 + 逐帧循环的独立实现。"""
    torch.manual_seed(3)
    rng = random.Random(7)
    for _ in range(25):
        total = rng.randint(4, 30)
        frame_len = rng.randint(1, min(8, total))
        hop_len = rng.randint(1, frame_len + 2)
        preemph = round(rng.uniform(0.0, 0.99), 3)
        window = rng.choice(["hann", "hamming", "rect"])
        wave = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(total)]
        got_y, got_frames = solve(torch.tensor(wave), preemph, frame_len, hop_len, window)
        want_y, want_frames = _ref(wave, preemph, frame_len, hop_len, window)
        assert got_y.shape == (total,), "y 长度不对"
        assert torch.allclose(got_y, torch.tensor(want_y), atol=1e-6), "随机用例 y 不一致"
        assert tuple(got_frames.shape) == (len(want_frames), frame_len), "帧矩阵形状不对"
        assert torch.allclose(got_frames, torch.tensor(want_frames), atol=1e-6), "随机用例 frames 不一致"


def check_invalid_inputs(solve):
    good = torch.randn(10)
    for bad in (
        lambda: solve(torch.randn(2, 5), 0.9, 4, 2),
        lambda: solve(good, 1.0, 4, 2),
        lambda: solve(good, -0.1, 4, 2),
        lambda: solve(good, 0.9, 0, 2),
        lambda: solve(good, 0.9, 4, 0),
        lambda: solve(good, 0.9, 11, 2),
        lambda: solve(good, 0.9, 4, 2, "blackman"),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
