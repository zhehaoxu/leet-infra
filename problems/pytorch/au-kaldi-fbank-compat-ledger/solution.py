"""Kaldi 兼容前端口径账（纯 Python float64）。

对应官方源码（pytorch/audio main 分支）：
  src/torchaudio/compliance/kaldi.py
  src/torchaudio/functional/functional.py::create_dct

五个 mode（全部显式接收参数，不依赖任何隐式状态 / RNG）：
  vtln_warp  —— VTLN 三段线性频率弯折（含官方四次赋值覆盖顺序）
  mel_banks  —— get_mel_banks 三角滤波器组（warp=1 min/max 分支 vs warp!=1 mask 分支）
  framing    —— _get_strided 分帧 + _feature_window_function 窗函数 + _get_window 前处理链
  fbank      —— 端到端 log-mel 滤波器组（use_power/use_log_fbank/use_energy/htk_compat/CMS）
  mfcc       —— DCT 矩阵 + lifter + htk_compat 尾部编排

数值口径：全程 Python float（float64）。
epsilon 恒为 torch.finfo(torch.float).eps = 1.1920928955078125e-07（官方 EPSILON
是 float32 标量，_get_epsilon 只做 .to(dtype)，数值不随 dtype 变化）。
"""

import cmath
import math

EPS = 1.1920928955078125e-07  # torch.finfo(torch.float).eps
WINDOW_TYPES = ["hamming", "hanning", "povey", "rectangular", "blackman"]


# ----------------------------------------------------------------- 基础工具
def _mel(freq):
    return 1127.0 * math.log(1.0 + freq / 700.0)


def _inverse_mel(mel_freq):
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def _log_energy(row, energy_floor):
    """官方 _get_log_energy：log(max(sum(x^2), EPS))，floor 在 log 域取 max。"""
    s = 0.0
    for v in row:
        s += v * v
    le = math.log(max(s, EPS))
    if energy_floor == 0.0:
        return le
    return max(le, math.log(energy_floor))


def _window_function(window_type, window_size, blackman_coeff):
    """官方 _feature_window_function 五分支（periodic=False）。"""
    if window_type not in WINDOW_TYPES:
        raise ValueError("Invalid window type %r" % (window_type,))
    n = window_size
    if window_type == "rectangular":
        return [1.0] * n
    if window_type in ("hanning", "hamming", "povey"):
        base = []
        for i in range(n):
            base.append(0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)))
        if window_type == "hamming":
            return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1))
                    for i in range(n)]
        if window_type == "povey":
            return [v ** 0.85 for v in base]
        return base
    # blackman：官方手写系数，不用 torch.blackman_window
    a = 2.0 * math.pi / (n - 1)
    out = []
    for i in range(n):
        out.append(blackman_coeff
                   - 0.5 * math.cos(a * i)
                   + (0.5 - blackman_coeff) * math.cos(2.0 * a * i))
    return out


def _get_strided(waveform, window_size, window_shift, snip_edges):
    """官方 _get_strided：snip_edges 两种分帧 + 反射填充。"""
    if window_shift <= 0:
        raise ValueError("window_shift must be positive")
    num_samples = len(waveform)
    if snip_edges:
        if num_samples < window_size:
            return []
        m = 1 + (num_samples - window_size) // window_shift
        padded = waveform
    else:
        reversed_waveform = waveform[::-1]
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            padded = pad_left + waveform + reversed_waveform
        else:
            padded = waveform[-pad:] + reversed_waveform
    frames = []
    for i in range(m):
        start = i * window_shift
        frames.append([float(padded[start + j]) for j in range(window_size)])
    return frames


def _get_window_py(waveform, window_size, window_shift, snip_edges,
                   window_type, blackman_coeff, dither, noise,
                   remove_dc_offset, preemphasis_coefficient,
                   raw_energy, energy_floor, padded_window_size):
    """官方 _get_window：dither → dc → raw_energy → preemphasis → window → pad → energy。"""
    if padded_window_size < window_size:
        raise ValueError("padded_window_size must be >= window_size")
    if padded_window_size % 2 != 0:
        raise ValueError("padded_window_size must be even")
    raw_frames = _get_strided(waveform, window_size, window_shift, snip_edges)
    m = len(raw_frames)
    strided = [list(row) for row in raw_frames]

    if dither != 0.0:
        if noise is None:
            raise ValueError("dither noise must be provided when dither != 0")
        if len(noise) != m or any(len(r) != window_size for r in noise):
            raise ValueError("dither noise shape mismatch")
        for i in range(m):
            for j in range(window_size):
                strided[i][j] += float(noise[i][j]) * dither

    if remove_dc_offset:
        for i in range(m):
            mu = sum(strided[i]) / window_size
            for j in range(window_size):
                strided[i][j] -= mu

    signal_log_energy = None
    if raw_energy:
        signal_log_energy = [_log_energy(row, energy_floor) for row in strided]

    if preemphasis_coefficient != 0.0:
        # F.pad(..., (1,0), mode='replicate')[:, :-1] ⇒ j=0 时取 x[0]
        for i in range(m):
            row = strided[i]
            new = [0.0] * window_size
            for j in range(window_size):
                prev = row[j - 1] if j > 0 else row[0]
                new[j] = row[j] - preemphasis_coefficient * prev
            strided[i] = new

    wf = _window_function(window_type, window_size, blackman_coeff)
    for i in range(m):
        for j in range(window_size):
            strided[i][j] *= wf[j]
        if padded_window_size != window_size:
            strided[i] += [0.0] * (padded_window_size - window_size)

    if not raw_energy:
        signal_log_energy = [_log_energy(row, energy_floor) for row in strided]

    return {
        "raw_frames": raw_frames,
        "strided": strided,
        "window_function": wf,
        "signal_log_energy": signal_log_energy,
        "num_frames": m,
        "epsilon": EPS,
        "padded_window_size": padded_window_size,
    }


# ----------------------------------------------------------------- VTLN
def vtln_warp_freq_py(vtln_low, vtln_high, low_freq, high_freq,
                      vtln_warp, freq):
    """官方 vtln_warp_freq：三段线性 + 四次赋值覆盖（顺序不可换）。"""
    if vtln_low <= low_freq:
        raise ValueError("vtln_low must be higher than low_freq")
    if vtln_high >= high_freq:
        raise ValueError("vtln_high must be lower than high_freq")
    if vtln_warp <= 0.0:
        raise ValueError("vtln_warp must be positive")
    l = vtln_low * max(1.0, vtln_warp)
    h = vtln_high * min(1.0, vtln_warp)
    scale = 1.0 / vtln_warp
    Fl = scale * l
    Fh = scale * h
    if not (l > low_freq and h < high_freq):
        raise ValueError("inflection points out of range")
    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = [0.0] * len(freq)
    # ① after_h：freq >= h
    for i, f in enumerate(freq):
        if f >= h:
            res[i] = high_freq + scale_right * (f - high_freq)
    # ② before_h：freq < h（覆盖 ① 的重叠部分）
    for i, f in enumerate(freq):
        if f < h:
            res[i] = scale * f
    # ③ before_l：freq < l（覆盖 ② 的重叠部分）
    for i, f in enumerate(freq):
        if f < l:
            res[i] = low_freq + scale_left * (f - low_freq)
    # ④ outside_low_high_freq：freq < low_freq 或 freq > high_freq（最后覆盖）
    for i, f in enumerate(freq):
        if f < low_freq or f > high_freq:
            res[i] = f
    return {
        "res": res,
        "l": l,
        "h": h,
        "scale": scale,
        "Fl": Fl,
        "Fh": Fh,
        "scale_left": scale_left,
        "scale_right": scale_right,
    }


def _warp_mel_py(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, mel_col):
    """官方 vtln_warp_mel_freq = mel_scale(vtln_warp_freq(inverse_mel_scale(mel)))。"""
    freq = [_inverse_mel(v[0]) if isinstance(v, (list, tuple)) else _inverse_mel(v)
            for v in mel_col]
    w = vtln_warp_freq_py(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, freq)
    return [[_mel(f)] for f in w["res"]]


# ----------------------------------------------------------------- mel banks
def get_mel_banks_py(num_bins, window_length_padded, sample_freq,
                     low_freq, high_freq, vtln_low, vtln_high, vtln_warp):
    """官方 get_mel_banks（返回 center_freqs 保持 (num_bins, 1) 形状）。"""
    if num_bins <= 3:
        raise ValueError("Must have at least 3 mel bins")
    if window_length_padded % 2 != 0:
        raise ValueError("window_length_padded must be even")
    num_fft_bins = int(window_length_padded / 2)
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    if not (0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist
            and low_freq < high_freq):
        raise ValueError("Bad values in options: low-freq / high-freq / nyquist")
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _mel(low_freq)
    mel_high_freq = _mel(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    if vtln_high < 0.0:
        vtln_high += nyquist
    if vtln_warp != 1.0:
        if not (low_freq < vtln_low < high_freq and 0.0 < vtln_high < high_freq
                and vtln_low < vtln_high):
            raise ValueError("Bad values in options: vtln-low / vtln-high")

    left_mel = [[mel_low_freq + bi * mel_freq_delta] for bi in range(num_bins)]
    center_mel = [[mel_low_freq + (bi + 1.0) * mel_freq_delta] for bi in range(num_bins)]
    right_mel = [[mel_low_freq + (bi + 2.0) * mel_freq_delta] for bi in range(num_bins)]
    if vtln_warp != 1.0:
        left_mel = _warp_mel_py(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, left_mel)
        center_mel = _warp_mel_py(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, center_mel)
        right_mel = _warp_mel_py(vtln_low, vtln_high, low_freq, high_freq, vtln_warp, right_mel)

    center_freqs = [[_inverse_mel(v[0])] for v in center_mel]
    mel = [_mel(fft_bin_width * k) for k in range(num_fft_bins)]

    bins = []
    for bi in range(num_bins):
        row = []
        lm, cm, rm = left_mel[bi][0], center_mel[bi][0], right_mel[bi][0]
        for k in range(num_fft_bins):
            up = (mel[k] - lm) / (cm - lm)
            down = (rm - mel[k]) / (rm - cm)
            if vtln_warp == 1.0:
                row.append(max(0.0, min(up, down)))
            else:
                if mel[k] > lm and mel[k] <= cm:
                    row.append(up)
                elif mel[k] > cm and mel[k] < rm:
                    row.append(down)
                else:
                    row.append(0.0)
        bins.append(row)
    return {
        "bins": bins,
        "center_freqs": center_freqs,
        "left_mel": left_mel,
        "center_mel": center_mel,
        "right_mel": right_mel,
        "mel_low_freq": mel_low_freq,
        "mel_high_freq": mel_high_freq,
        "mel_freq_delta": mel_freq_delta,
        "num_fft_bins": num_fft_bins,
        "nyquist": nyquist,
        "fft_bin_width": fft_bin_width,
        "high_freq_eff": high_freq,
        "vtln_high_eff": vtln_high,
    }


# ----------------------------------------------------------------- FFT
def _rfft_abs(row):
    """纯 Python DFT 取 k=0..N/2 的 |X[k]|（等价 torch.fft.rfft(x).abs()）。"""
    n = len(row)
    out = []
    for k in range(n // 2 + 1):
        acc = 0j
        for t in range(n):
            acc += row[t] * cmath.exp(-2j * math.pi * k * t / n)
        out.append(abs(acc))
    return out


def _subtract_column_mean(mat, subtract_mean):
    if not subtract_mean or not mat:
        return mat
    m, n = len(mat), len(mat[0])
    means = [sum(mat[i][j] for i in range(m)) / m for j in range(n)]
    return [[mat[i][j] - means[j] for j in range(n)] for i in range(m)]


_FRAMING_DEFAULTS = {
    "snip_edges": True,
    "window_type": "povey",
    "blackman_coeff": 0.42,
    "dither": 0.0,
    "noise": None,
    "remove_dc_offset": True,
    "preemphasis_coefficient": 0.97,
    "raw_energy": True,
    "energy_floor": 1.0,
}


def _framing_kwargs(kw):
    keys = ("window_size", "window_shift", "snip_edges", "window_type",
            "blackman_coeff", "dither", "noise", "remove_dc_offset",
            "preemphasis_coefficient", "raw_energy", "energy_floor",
            "padded_window_size")
    fk = {}
    for k in keys:
        if k in kw:
            fk[k] = kw[k]
        elif k in _FRAMING_DEFAULTS:
            fk[k] = _FRAMING_DEFAULTS[k]
        else:
            raise ValueError("framing params missing: %s" % (k,))
    return fk


def _fbank_py(waveform, fk, sample_freq, num_mel_bins, low_freq, high_freq,
              vtln_low, vtln_high, vtln_warp,
              use_power, use_log_fbank, use_energy, htk_compat,
              subtract_mean):
    win = _get_window_py(waveform, **fk)
    frames = win["strided"]
    padded = fk["padded_window_size"]
    banks = get_mel_banks_py(num_mel_bins, padded, sample_freq,
                             low_freq, high_freq, vtln_low, vtln_high, vtln_warp)
    bins = banks["bins"]
    # pad right column with zeros ⇒ (num_bins, num_fft_bins + 1)
    bins_padded = [list(row) + [0.0] for row in bins]

    spectrum = []
    for row in frames:
        sp = _rfft_abs(row)
        if use_power:
            sp = [v * v for v in sp]
        spectrum.append(sp)

    mel_energies = []
    for row in spectrum:
        out = [0.0] * num_mel_bins
        for bi in range(num_mel_bins):
            s = 0.0
            for k in range(len(row)):
                s += row[k] * bins_padded[bi][k]
            out[bi] = s
        mel_energies.append(out)

    if use_log_fbank:
        mel_energies = [[math.log(max(v, EPS)) for v in row]
                        for row in mel_energies]

    features = mel_energies
    if use_energy:
        sig = [[v] for v in win["signal_log_energy"]]
        if htk_compat:
            features = [features[i] + sig[i] for i in range(len(features))]
        else:
            features = [sig[i] + features[i] for i in range(len(features))]
    features = _subtract_column_mean(features, subtract_mean)
    return {
        "features": features,
        "bins": bins,
        "bins_padded": bins_padded,
        "spectrum": spectrum,
        "mel_energies": mel_energies,
        "signal_log_energy": win["signal_log_energy"],
        "center_freqs": banks["center_freqs"],
        "raw_frames": win["raw_frames"],
        "strided": win["strided"],
        "window_function": win["window_function"],
    }


# ----------------------------------------------------------------- MFCC
def _dct_matrix_py(num_ceps, num_mel_bins):
    """官方 _get_dct_matrix：create_dct(...,'ortho') 转置 + 首列强制 sqrt(1/n)。"""
    n = num_mel_bins
    mat = [[0.0] * n for _ in range(n)]  # (n_mels, n_mfcc)，转置后
    for k in range(n):
        for j in range(n):
            v = math.cos(math.pi / n * (j + 0.5) * k)
            if k == 0:
                v *= 1.0 / math.sqrt(2.0)
            v *= math.sqrt(2.0 / n)
            mat[j][k] = v
    for j in range(n):
        mat[j][0] = math.sqrt(1.0 / n)
    return [row[:num_ceps] for row in mat]


def _lifter_coeffs_py(num_ceps, cepstral_lifter):
    return [1.0 + 0.5 * cepstral_lifter
            * math.sin(math.pi * i / cepstral_lifter)
            for i in range(num_ceps)]


def _mfcc_py(waveform, fk, sample_freq, num_mel_bins, low_freq, high_freq,
             vtln_low, vtln_high, vtln_warp,
             num_ceps, cepstral_lifter, use_energy, htk_compat,
             subtract_mean):
    if num_ceps > num_mel_bins:
        raise ValueError("num_ceps cannot be larger than num_mel_bins: %d vs %d"
                         % (num_ceps, num_mel_bins))
    # mfcc 强制 fbank：subtract_mean=False, use_log_fbank=True, use_power=True
    fb = _fbank_py(waveform, fk, sample_freq, num_mel_bins, low_freq, high_freq,
                   vtln_low, vtln_high, vtln_warp,
                   True, True, use_energy, htk_compat, False)
    feature = [list(row) for row in fb["features"]]

    signal_log_energy = None
    if use_energy:
        col = num_mel_bins if htk_compat else 0
        signal_log_energy = [row[col] for row in feature]
        mel_offset = 0 if htk_compat else 1
        feature = [row[mel_offset:num_mel_bins + mel_offset] for row in feature]

    dct = _dct_matrix_py(num_ceps, num_mel_bins)
    out = []
    for row in feature:
        new = [0.0] * num_ceps
        for c in range(num_ceps):
            s = 0.0
            for j in range(num_mel_bins):
                s += row[j] * dct[j][c]
            new[c] = s
        out.append(new)

    lifter = [1.0] * num_ceps
    if cepstral_lifter != 0.0:
        lifter = _lifter_coeffs_py(num_ceps, cepstral_lifter)
        for row in out:
            for c in range(num_ceps):
                row[c] *= lifter[c]

    if use_energy:
        for i in range(len(out)):
            out[i][0] = signal_log_energy[i]

    if htk_compat:
        energy = [[row[0] if use_energy else row[0] * math.sqrt(2.0)]
                  for row in out]
        out = [row[1:] + energy[i] for i, row in enumerate(out)]

    out = _subtract_column_mean(out, subtract_mean)
    return {
        "features": out,
        "dct_matrix": dct,
        "lifter_coeffs": lifter,
        "fbank_features": fb["features"],
    }


# ----------------------------------------------------------------- solve
def solve(mode, **kw):
    if mode == "vtln_warp":
        return vtln_warp_freq_py(
            kw["vtln_low"], kw["vtln_high"], kw["low_freq"],
            kw["high_freq"], kw["vtln_warp"], kw["freq"])
    if mode == "mel_banks":
        return get_mel_banks_py(
            kw["num_bins"], kw["window_length_padded"], kw["sample_freq"],
            kw["low_freq"], kw["high_freq"], kw["vtln_low"],
            kw["vtln_high"], kw["vtln_warp"])
    if mode == "framing":
        fk = _framing_kwargs(kw)
        return _get_window_py(kw["waveform"], **fk)
    if mode == "fbank":
        fk = _framing_kwargs(kw)
        return _fbank_py(
            kw["waveform"], fk, kw["sample_freq"], kw["num_mel_bins"],
            kw["low_freq"], kw["high_freq"], kw["vtln_low"],
            kw["vtln_high"], kw["vtln_warp"], kw["use_power"],
            kw["use_log_fbank"], kw["use_energy"], kw["htk_compat"],
            kw["subtract_mean"])
    if mode == "mfcc":
        fk = _framing_kwargs(kw)
        return _mfcc_py(
            kw["waveform"], fk, kw["sample_freq"], kw["num_mel_bins"],
            kw["low_freq"], kw["high_freq"], kw["vtln_low"],
            kw["vtln_high"], kw["vtln_warp"], kw["num_ceps"],
            kw["cepstral_lifter"], kw["use_energy"], kw["htk_compat"],
            kw["subtract_mean"])
    raise ValueError("unknown mode: %r" % (mode,))
