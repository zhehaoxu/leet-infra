import math

import torch

_LOGSTEP = math.log(6.4) / 27.0
_MEL_KNEE = 15.0  # Slaney: 1000 Hz 处的 mel 值 = 1000 / (200/3)


def _hz_to_mel(f, scale):
    if scale == "htk":
        return 2595.0 * math.log10(1.0 + f / 700.0)
    if f < 1000.0:
        return f / (200.0 / 3.0)
    return _MEL_KNEE + math.log(f / 1000.0) / _LOGSTEP


def _mel_to_hz(m, scale):
    if scale == "htk":
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    if m < _MEL_KNEE:
        return m * (200.0 / 3.0)
    return 1000.0 * math.exp((m - _MEL_KNEE) * _LOGSTEP)


def solve(power_spec, sample_rate, n_mels, f_min, f_max, scale):
    if not isinstance(power_spec, torch.Tensor) or power_spec.dim() != 2:
        raise ValueError("power_spec 必须是二维张量")
    if scale not in ("htk", "slaney"):
        raise ValueError("scale 必须是 htk 或 slaney")
    if n_mels < 1:
        raise ValueError("n_mels 必须为正")
    if not (0.0 <= float(f_min) < float(f_max) <= sample_rate / 2.0):
        raise ValueError("需要 0 <= f_min < f_max <= sample_rate/2")

    n_freqs = int(power_spec.size(1))
    if n_freqs < 2:
        raise ValueError("功率谱至少要 2 个频点")

    # mel 域等距 -> Hz 边界；两两严格递增才谈得上三角滤波器
    m_lo, m_hi = _hz_to_mel(float(f_min), scale), _hz_to_mel(float(f_max), scale)
    edges = [
        _mel_to_hz(m_lo + (m_hi - m_lo) * i / (n_mels + 1), scale)
        for i in range(n_mels + 2)
    ]
    for i in range(n_mels + 1):
        if not edges[i + 1] > edges[i]:
            raise ValueError("Mel 边界点在 Hz 上不严格递增（n_mels 相对频带过密）")

    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=torch.float64)
    filters = torch.zeros((n_mels, n_freqs), dtype=torch.float64)
    for m in range(n_mels):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        rise = (freqs - lo) / (ce - lo)
        fall = (hi - freqs) / (hi - ce)
        w = torch.clamp(torch.minimum(rise, fall), min=0.0)
        if scale == "slaney":
            w = w * (2.0 / (hi - lo))
        filters[m] = w

    mel = power_spec.to(torch.float64) @ filters.T
    logmel = torch.log(torch.clamp(mel, min=1e-10))
    return (
        filters.to(torch.float32),
        mel.to(torch.float32),
        logmel.to(torch.float32),
    )
