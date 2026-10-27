import torch


def solve(wave, n_fft, hop_len, center):
    """短时傅里叶变换的幅度谱（Hann 窗、周期型口径）。

    wave:(T,) 单声道波形；n_fft:FFT 点数（偶数）；hop_len:帧移（1 <= hop_len <= n_fft）；
    center:True 时按 torch.stft 语义在两侧各补 n_fft//2 个**反射**样本（不重复边界点，
    首尾各镜像 pad 个点），False 时不补。

    帧数：center=True -> F = 1 + T // hop_len；center=False -> F = 1 + (T - n_fft) // hop_len。
    窗：w[n] = 0.5 - 0.5*cos(2*pi*n/n_fft)，n = 0..n_fft-1（**周期型**，长度 n_fft）。
    输出 magnitude = |rfft(frame * w, n_fft)|，形状 (F, n_fft//2 + 1)，float32。
    禁止直接调用 torch.stft / torchaudio 的封装。
    """
    # TODO
    raise NotImplementedError
