import torch


def solve(power_spec, sample_rate, n_mels, f_min, f_max, scale):
    """构造三角 Mel 滤波器组，并把功率谱投影成 mel / log-mel。

    power_spec:(T, F) 线性功率谱（F 个频点，等间隔铺在 [0, sample_rate/2]）；
    sample_rate:采样率;n_mels:滤波器个数;f_min / f_max:频率范围（Hz）；
    scale:"htk" 或 "slaney"——两套频率↔mel 映射。

    1) 在 mel 域取 n_mels + 2 个等距边界点（含 f_min、f_max），再映射回 Hz；
    2) 第 m 个三角滤波器在 [hz[m], hz[m+1], hz[m+2]] 上先升后降，其余为 0；
    3) scale == "slaney" 时每个滤波器再乘面积归一 2 / (hz[m+2] - hz[m])；
       scale == "htk" 时不归一（HTK 口径）；
    4) mel = power_spec @ filters.T，logmel = log(max(mel, 1e-10))。

    HTK 刻度：mel = 2595 * log10(1 + f/700)；
    Slaney 刻度：f < 1000 时 mel = f / (200/3)，否则 mel = 15 + ln(f/1000) / (ln(6.4)/27)。

    返回 (filters:(n_mels, F) float32, mel:(T, n_mels) float32, logmel:(T, n_mels) float32)。
    """
    # TODO
    raise NotImplementedError
