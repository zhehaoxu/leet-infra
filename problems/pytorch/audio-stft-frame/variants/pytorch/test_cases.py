import torch


def _oracle(wave, n_fft, hop_len, center):
    """独立 oracle：直接调 torch.stft（Hann 周期窗 + 反射填充）。"""
    ref = torch.stft(
        wave,
        n_fft,
        hop_length=hop_len,
        window=torch.hann_window(n_fft, dtype=wave.dtype),
        center=center,
        pad_mode="reflect",
        return_complex=True,
    )
    return ref.abs().T


def check_example(solve):
    torch.manual_seed(0)
    wave = torch.arange(1, 10, dtype=torch.float32) / 10.0
    mag = solve(wave, 4, 2, True)
    want = torch.tensor(
        [
            [0.3, 0.1, 0.1],
            [0.6, 0.316228, 0.0],
            [1.0, 0.509902, 0.0],
            [1.4, 0.707107, 0.0],
            [1.7, 0.9, 0.1],
        ]
    )
    assert mag.shape == (5, 3), f"应为 5 帧 × 3 个频点，got {tuple(mag.shape)}"
    assert torch.allclose(mag, want, atol=1e-5), f"幅度谱不对：{mag.tolist()}"


check_example.example = True


def check_matches_torch_stft(solve):
    """两种 center 设置都应与 torch.stft 逐元素一致。"""
    torch.manual_seed(1)
    for center in (True, False):
        for n_fft, hop in ((8, 4), (16, 5), (4, 4)):
            wave = torch.randn(40)
            got = solve(wave, n_fft, hop, center)
            want = _oracle(wave, n_fft, hop, center)
            assert got.shape == want.shape, f"shape {tuple(got.shape)} vs {tuple(want.shape)}"
            assert torch.allclose(got, want, atol=1e-5), f"center={center} n_fft={n_fft} 不一致"


def check_center_equivalence(solve):
    """center=True 只是多补了 n_fft//2 的两侧反射帧，中间帧与 center=False 相同。"""
    torch.manual_seed(2)
    wave = torch.randn(16)
    n_fft, hop = 4, 2
    a = solve(wave, n_fft, hop, True)
    b = solve(wave, n_fft, hop, False)
    assert a.shape[0] == 1 + 16 // hop, f"center=True 帧数应为 9，got {a.shape[0]}"
    assert b.shape[0] == 1 + (16 - n_fft) // hop, f"center=False 帧数应为 7，got {b.shape[0]}"
    assert torch.allclose(a[1 : 1 + b.shape[0]], b, atol=1e-5), "center 模式的中间帧应完全一致"

    # 常数信号乘周期 Hann(N)：直流 = amp*N/2，两侧邻频各 = amp*N/4，其余为 0
    const = solve(torch.ones(12) * 2.0, n_fft, hop, True)
    assert const.shape[0] == 7, f"帧数不对：{const.shape[0]}"
    assert torch.allclose(const[:, 0], torch.full((7,), 4.0), atol=1e-5), "直流频点应为 2*4/2=4"
    assert torch.allclose(const[:, 1], torch.full((7,), 2.0), atol=1e-5), "邻频应为 2*4/4=2"
    assert torch.all(const[:, 2:].abs() < 1e-5), "Hann 窗声常数信号不应有更高频能量"


def check_pure_tone_bin(solve):
    """整周期正弦只激励对应的频点，幅度为 n_fft/2 * 窗增益。"""
    torch.manual_seed(3)
    n_fft, hop = 16, 8
    k = 4
    n = torch.arange(n_fft, dtype=torch.float32)
    wave = torch.sin(2 * torch.pi * k * n / n_fft).repeat(3)
    mag = solve(wave, n_fft, hop, False)
    # 每一帧都是整周期正弦
    assert torch.all(mag.amax(dim=1) > 1.0), "整周期正弦的峰值频点应显著非零"
    peaks = mag.argmax(dim=1)
    assert torch.all(peaks == k), f"峰值应落在第 {k} 个频点，got {peaks.tolist()}"
    # Hann 窗把能量铺到 k-1 / k / k+1 三个频点，其余频点严格为 0
    keep = mag.clone()
    keep[:, k - 1 : k + 2] = 0.0
    assert torch.all(keep.abs() < 1e-4), f"窗泄漏之外的频点应为 0，got {keep.abs().amax()}"


def check_invalid_inputs(solve):
    good = torch.randn(20)
    for bad in (
        lambda: solve(torch.randn(4, 5), 8, 4, True),
        lambda: solve(good, 7, 4, True),
        lambda: solve(good, 0, 4, True),
        lambda: solve(good, 8, 0, True),
        lambda: solve(good, 8, 16, True),
        lambda: solve(torch.randn(5), 8, 4, False),
        lambda: solve(torch.randn(4), 8, 4, True),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
