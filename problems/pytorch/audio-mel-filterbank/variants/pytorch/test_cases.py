import math

import torch


def _hz_to_mel(f, scale):
    if scale == "htk":
        return 2595.0 * math.log10(1.0 + f / 700.0)
    if f < 1000.0:
        return f / (200.0 / 3.0)
    return 15.0 + math.log(f / 1000.0) / (math.log(6.4) / 27.0)


def _mel_to_hz(m, scale):
    if scale == "htk":
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    if m < 15.0:
        return m * (200.0 / 3.0)
    return 1000.0 * math.exp((m - 15.0) * (math.log(6.4) / 27.0))


def _oracle(power, sr, n_mels, f_min, f_max, scale):
    """独立 oracle：逐 (滤波器, 频点) 用分支判断算左右斜坡，不做向量化。"""
    rows, n_freqs = len(power), len(power[0])
    freqs = [sr / 2.0 * i / (n_freqs - 1) for i in range(n_freqs)]
    lo_mel, hi_mel = _hz_to_mel(f_min, scale), _hz_to_mel(f_max, scale)
    edges = [
        _mel_to_hz(lo_mel + (hi_mel - lo_mel) * i / (n_mels + 1), scale)
        for i in range(n_mels + 2)
    ]
    filters = []
    for m in range(n_mels):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        row = []
        for f in freqs:
            if f <= lo or f >= hi:
                w = 0.0
            elif f <= ce:
                w = (f - lo) / (ce - lo)
            else:
                w = (hi - f) / (hi - ce)
            if scale == "slaney":
                w *= 2.0 / (hi - lo)
            row.append(w)
        filters.append(row)
    mel = [
        [sum(power[t][k] * filters[m][k] for k in range(n_freqs)) for m in range(n_mels)]
        for t in range(rows)
    ]
    logmel = [[math.log(v) if v > 1e-10 else math.log(1e-10) for v in row] for row in mel]
    return filters, mel, logmel


_POWER = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
]


def check_example(solve):
    """8 kHz 采样、9 个频点、3 个 HTK Mel 滤波器。"""
    torch.manual_seed(0)
    filters, mel, logmel = solve(torch.tensor(_POWER), 8000, 3, 0.0, 4000.0, "htk")

    assert tuple(filters.shape) == (3, 9), f"filters 形状应为 (3, 9)，got {tuple(filters.shape)}"
    # 三角滤波器在中心频点附近取到 1，两侧线性衰减
    assert abs(float(filters[0, 1]) - 0.893459) < 1e-5, f"got {float(filters[0, 1])}"
    assert abs(float(filters[1, 2]) - 0.834308) < 1e-5, f"got {float(filters[1, 2])}"
    assert abs(float(filters[2, 4]) - 0.801285) < 1e-5, f"got {float(filters[2, 4])}"

    want_mel = torch.tensor([[2.283994, 6.312877, 16.6376], [0.529576, 0.895194, 1.417816]])
    assert torch.allclose(mel, want_mel, atol=1e-5), f"mel 不对：{mel.tolist()}"
    want_log = torch.tensor([[0.825926, 1.842592, 2.811665], [-0.63568, -0.110715, 0.349118]])
    assert torch.allclose(logmel, want_log, atol=1e-5), f"log-mel 不对：{logmel.tolist()}"


check_example.example = True


def check_slaney_scale(solve):
    """同样的输入换成 Slaney 刻度 + 面积归一，能量分布完全不同。"""
    torch.manual_seed(1)
    filters, mel, logmel = solve(torch.tensor(_POWER), 8000, 3, 0.0, 4000.0, "slaney")
    # 面积归一 = 2 / (高频边 - 低频边)，把 4000 Hz 量级的带宽压到 ~1e-3
    assert abs(float(filters[1, 3]) - 0.000865) < 1e-6, f"got {float(filters[1, 3])}"
    want_mel = torch.tensor([[0.004462, 0.007182, 0.011634], [0.000982, 0.000975, 0.000989]])
    assert torch.allclose(mel, want_mel, atol=1e-6), f"slaney mel 不对：{mel.tolist()}"
    # 线性功率谱下 Slaney 把更多能量分给低频带，HTK 则相反
    _, mel_htk, _ = solve(torch.tensor(_POWER), 8000, 3, 0.0, 4000.0, "htk")
    assert float(mel[0, 0]) < float(mel_htk[0, 0]), "Slaney 归一后高频带的绝对值更小"
    assert abs(float(logmel[0, 0]) + 5.412123) < 1e-5, f"got {float(logmel[0, 0])}"


def check_tone_localization(solve):
    """单频点功率谱：响应集中在覆盖该频率的滤波器上，峰值下标随频率单调不减。"""
    torch.manual_seed(2)
    n_freqs = 17
    prev = -1
    for k in range(2, n_freqs):
        power = torch.zeros(1, n_freqs)
        power[0, k] = 1.0
        filters, mel, _ = solve(power, 16000, 4, 0.0, 8000.0, "htk")
        assert torch.allclose(mel[0], filters[:, k], atol=1e-6), "单频点响应应等于滤波器列"
        peak = int(mel[0].argmax())
        assert peak >= prev, f"频点 {k} 的峰值带下标 {peak} 小于前面的 {prev}"
        prev = peak
    assert prev == 3, f"最高频点应落在最后一个滤波器，got {prev}"

    # 直流频点位于第一个滤波器的左端点上，权重为 0 -> 全零 -> 落到 log 下限
    power = torch.zeros(1, n_freqs)
    power[0, 0] = 5.0
    _, mel, logmel = solve(power, 16000, 4, 0.0, 8000.0, "htk")
    assert float(mel.abs().max()) == 0.0, "DC 频点在三角滤波器端点，响应应为 0"
    assert torch.allclose(logmel, torch.full((1, 4), math.log(1e-10)), atol=1e-5), "应落到 log 下限"


def check_random_vs_oracle(solve):
    """随机功率谱 + 随机配置对拍逐点分支的独立实现。"""
    torch.manual_seed(3)
    for seed in range(12):
        g = torch.Generator().manual_seed(seed)
        n_freqs = int(torch.randint(5, 12, (1,), generator=g))
        n_mels = int(torch.randint(1, 4, (1,), generator=g))
        sr = int(torch.randint(8, 24, (1,), generator=g)) * 1000
        power = torch.rand((2, n_freqs), generator=g)
        for scale in ("htk", "slaney"):
            f_max = sr / 2.0
            got_f, got_m, got_l = solve(power, sr, n_mels, 0.0, f_max, scale)
            want_f, want_m, want_l = _oracle(power.tolist(), sr, n_mels, 0.0, f_max, scale)
            assert torch.allclose(got_f, torch.tensor(want_f), atol=1e-6), "filters 不一致"
            assert torch.allclose(got_m, torch.tensor(want_m), atol=1e-5), "mel 不一致"
            assert torch.allclose(got_l, torch.tensor(want_l), atol=1e-4), "logmel 不一致"


def check_offband_zero_and_invalid(solve):
    """频带外的频点严格为 0；非法配置抛 ValueError。"""
    torch.manual_seed(4)
    power = torch.ones(1, 9)
    filters, _, _ = solve(power, 8000, 3, 500.0, 3000.0, "htk")
    freqs = torch.linspace(0.0, 4000.0, 9)
    outside = (freqs < 500.0) | (freqs > 3000.0)
    assert torch.all(filters[:, outside].abs() < 1e-7), "频带外不应有响应"
    assert torch.all(filters[:, ~outside].sum(dim=1) > 0.0), "频带内每个滤波器都要有响应"

    good = torch.ones(1, 9)
    for bad in (
        lambda: solve(good, 8000, 3, 0.0, 4000.0, "mel"),
        lambda: solve(good[:, 0], 8000, 3, 0.0, 4000.0, "htk"),
        lambda: solve(good, 8000, 0, 0.0, 4000.0, "htk"),
        lambda: solve(good, 8000, 3, 4000.0, 4000.0, "htk"),
        lambda: solve(good, 8000, 3, -1.0, 4000.0, "htk"),
        lambda: solve(good, 8000, 3, 0.0, 4100.0, "htk"),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
