"""852 判题用例：Kaldi 兼容前端口径账。

约定：顶层 check_*(solve) 由 judge 发现并执行，solve 由 judge 注入；
本文件不得 import solution。每个 check 自带独立 oracle（纯 Python），
并含双向断言（钉死官方 quirk，防止朴素实现蒙混）与域约束消息断言。
"""

import cmath
import math

EPS = 1.1920928955078125e-07


# ----------------------------------------------------------------- 通用工具
class _LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return (self.s / 2147483647.0) * 2.0 - 1.0


def _wave(n, seed):
    g = _LCG(seed)
    return [round(g.next() * 3.0 + 1.0, 6) for _ in range(n)]


def _noise(m, n, seed):
    g = _LCG(seed)
    return [[g.next() for _ in range(n)] for _ in range(m)]


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [v for it in x for v in _flat(it)]
    return [x]


def _eq(a, b, tol=1e-8):
    if a is None or b is None or isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                return False
        elif abs(float(x) - float(y)) > tol:
            return False
    return True


def _must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _must_differ(a, b, msg):
    _must(not _eq(a, b), msg + " (two candidates unexpectedly equal)")


def _raises(fn, keyword, msg):
    try:
        fn()
    except Exception as e:
        _must(keyword in str(e), "%s: message %r lacks keyword %r"
              % (msg, str(e), keyword))
        return
    raise AssertionError("%s: expected ValueError containing %r, none raised"
                         % (msg, keyword))


# ----------------------------------------------------------------- 独立 oracle
def _mel(f):
    return 1127.0 * math.log(1.0 + f / 700.0)


def _imel(m):
    return 700.0 * math.exp(m / 1127.0) - 700.0


def _o_vtln(vl, vh, lo, hi, wp, freq):
    l, h = vl * max(1.0, wp), vh * min(1.0, wp)
    sc = 1.0 / wp
    sl = (sc * l - lo) / (l - lo)
    sr = (hi - sc * h) / (hi - h)
    out = []
    for f in freq:
        if f < lo or f > hi:
            out.append(f)
        elif f < l:
            out.append(lo + sl * (f - lo))
        elif f < h:
            out.append(sc * f)
        else:
            out.append(hi + sr * (f - hi))
    return dict(res=out, l=l, h=h, scale=sc, Fl=sc * l, Fh=sc * h,
                scale_left=sl, scale_right=sr)


def _o_banks(nb, pad, sf, lo, hi, vl, vh, wp):
    nyq = sf / 2.0
    hi = hi + nyq if hi <= 0 else hi
    vh = vh + nyq if vh < 0 else vh
    fbw = sf / pad
    ml, mh = _mel(lo), _mel(hi)
    delta = (mh - ml) / (nb + 1)
    nfft = pad // 2
    grid = [_mel(fbw * k) for k in range(nfft)]
    bins, cf = [], []
    for b in range(nb):
        lm = ml + b * delta
        cm = ml + (b + 1.0) * delta
        rm = ml + (b + 2.0) * delta
        if wp != 1.0:
            lm = _mel(_o_vtln(vl, vh, lo, hi, wp, [_imel(lm)])["res"][0])
            cm = _mel(_o_vtln(vl, vh, lo, hi, wp, [_imel(cm)])["res"][0])
            rm = _mel(_o_vtln(vl, vh, lo, hi, wp, [_imel(rm)])["res"][0])
        cf.append([_imel(cm)])
        row = []
        for g in grid:
            up, dn = (g - lm) / (cm - lm), (rm - g) / (rm - cm)
            if wp == 1.0:
                row.append(max(0.0, min(up, dn)))
            elif g > lm and g <= cm:
                row.append(up)
            elif g > cm and g < rm:
                row.append(dn)
            else:
                row.append(0.0)
        bins.append(row)
    return dict(bins=bins, center_freqs=cf, mel_low_freq=ml,
                mel_high_freq=mh, mel_freq_delta=delta,
                num_fft_bins=nfft, nyquist=nyq, fft_bin_width=fbw,
                high_freq_eff=hi, vtln_high_eff=vh)


def _o_window(wt, n, coeff):
    if wt == "rectangular":
        return [1.0] * n
    s = [math.sin(math.pi * i / (n - 1)) ** 2 for i in range(n)]
    if wt == "hanning":
        return s
    if wt == "povey":
        return [v ** 0.85 for v in s]
    if wt == "hamming":
        return [0.08 + 0.92 * v for v in s]
    a = 2.0 * math.pi / (n - 1)
    return [coeff - 0.5 * math.cos(a * i) + (0.5 - coeff) * math.cos(2 * a * i)
            for i in range(n)]


def _o_strided(w, ws, shift, snip):
    n = len(w)
    if snip:
        if n < ws:
            return []
        m = 1 + (n - ws) // shift
        buf = w
    else:
        rev = w[::-1]
        m = (n + shift // 2) // shift
        pad = ws // 2 - shift // 2
        buf = (rev[-pad:] + w + rev) if pad > 0 else (w[-pad:] + rev)
    return [[buf[i * shift + j] for j in range(ws)] for i in range(m)]


def _o_log_energy(row, floor):
    s = sum(v * v for v in row)
    le = math.log(max(s, EPS))
    return le if floor == 0.0 else max(le, math.log(floor))


def _o_chain(w, fr):
    ws = fr["window_size"]
    shift = fr["window_shift"]
    snip = fr["snip_edges"]
    wt = fr["window_type"]
    coeff = fr["blackman_coeff"]
    dither = fr["dither"]
    noise = fr["noise"]
    dc = fr["remove_dc_offset"]
    pre = fr["preemphasis_coefficient"]
    raw = fr["raw_energy"]
    floor = fr["energy_floor"]
    padded = fr["padded_window_size"]
    rawf = _o_strided(w, ws, shift, snip)
    rows = [list(r) for r in rawf]
    if dither != 0.0:
        rows = [[rows[i][j] + noise[i][j] * dither for j in range(ws)]
                for i in range(len(rows))]
    if dc:
        rows = [[v - sum(r) / ws for v in r] for r in rows]
    sig = [_o_log_energy(r, floor) for r in rows] if raw else None
    if pre != 0.0:
        rows = [[r[j] - pre * (r[j - 1] if j else r[0]) for j in range(ws)]
                for r in rows]
    wf = _o_window(wt, ws, coeff)
    rows = [[r[j] * wf[j] for j in range(ws)]
            + [0.0] * (padded - ws) for r in rows]
    if not raw:
        sig = [_o_log_energy(r, floor) for r in rows]
    return dict(raw_frames=rawf, strided=rows, window_function=wf,
                signal_log_energy=sig, num_frames=len(rows),
                epsilon=EPS, padded_window_size=padded)


def _o_dft(row):
    n = len(row)
    return [abs(sum(row[t] * cmath.exp(-2j * math.pi * k * t / n)
                    for t in range(n))) for k in range(n // 2 + 1)]


def _o_cms(mat, flag):
    if not flag or not mat:
        return mat
    n = len(mat[0])
    means = [sum(r[j] for r in mat) / len(mat) for j in range(n)]
    return [[r[j] - means[j] for j in range(n)] for r in mat]


def _o_fbank(w, fr, sf, nb, lo, hi, vl, vh, wp, up, ul, ue, htk, sm):
    ch = _o_chain(w, fr)
    bk = _o_banks(nb, fr["padded_window_size"], sf, lo, hi, vl, vh, wp)
    padded = [r + [0.0] for r in bk["bins"]]
    sp = []
    for r in ch["strided"]:
        a = _o_dft(r)
        sp.append([v * v for v in a] if up else a)
    mel = [[sum(sp[i][k] * padded[b][k] for k in range(len(sp[i])))
            for b in range(nb)] for i in range(len(sp))]
    if ul:
        mel = [[math.log(max(v, EPS)) for v in r] for r in mel]
    feats = mel
    if ue:
        feats = [(feats[i] + [ch["signal_log_energy"][i]]) if htk
                 else ([ch["signal_log_energy"][i]] + feats[i])
                 for i in range(len(feats))]
    feats = _o_cms(feats, sm)
    return dict(features=feats, bins=bk["bins"], bins_padded=padded,
                spectrum=sp, mel_energies=mel,
                signal_log_energy=ch["signal_log_energy"],
                center_freqs=bk["center_freqs"], raw_frames=ch["raw_frames"],
                strided=ch["strided"], window_function=ch["window_function"])


def _o_dct(nc, nb):
    n = nb
    m = [[math.cos(math.pi / n * (j + 0.5) * k) * math.sqrt(2.0 / n)
          * (1.0 / math.sqrt(2.0) if k == 0 else 1.0) for k in range(n)]
         for j in range(n)]
    for j in range(n):
        m[j][0] = math.sqrt(1.0 / n)
    return [r[:nc] for r in m]


def _o_mfcc(w, fr, sf, nb, lo, hi, vl, vh, wp, nc, lif, ue, htk, sm):
    fb = _o_fbank(w, fr, sf, nb, lo, hi, vl, vh, wp, True, True, ue, htk, False)
    rows = [list(r) for r in fb["features"]]
    sig = None
    if ue:
        col = nb if htk else 0
        sig = [r[col] for r in rows]
        off = 0 if htk else 1
        rows = [r[off:nb + off] for r in rows]
    dct = _o_dct(nc, nb)
    out = [[sum(r[j] * dct[j][c] for j in range(nb)) for c in range(nc)]
           for r in rows]
    lc = [1.0] * nc
    if lif != 0.0:
        lc = [1.0 + 0.5 * lif * math.sin(math.pi * i / lif)
              for i in range(nc)]
        out = [[r[c] * lc[c] for c in range(nc)] for r in out]
    if ue:
        for i in range(len(out)):
            out[i][0] = sig[i]
    if htk:
        en = [[r[0] * (1.0 if ue else math.sqrt(2.0))] for r in out]
        out = [r[1:] + en[i] for i, r in enumerate(out)]
    out = _o_cms(out, sm)
    return dict(features=out, dct_matrix=dct, lifter_coeffs=lc,
                fbank_features=fb["features"])


def _fr(**over):
    d = dict(window_size=32, window_shift=16, snip_edges=True,
             window_type="hamming", blackman_coeff=0.42, dither=0.0,
             noise=None, remove_dc_offset=True,
             preemphasis_coefficient=0.97, raw_energy=True,
             energy_floor=1.0, padded_window_size=32)
    d.update(over)
    return d


SF = 16000.0


# =========================================================================
# check 1：VTLN 三段线性
# =========================================================================
def check_vtln_warp(solve):
    cases = [
        (100.0, 7500.0, 20.0, 8000.0, 0.9),
        (100.0, 7500.0, 20.0, 8000.0, 1.0),
        (100.0, 7500.0, 20.0, 8000.0, 1.2),
        (200.0, 6000.0, 40.0, 7000.0, 0.8),
        (7000.0, 100.0, 20.0, 8000.0, 2.0),  # l > h，覆盖顺序生效
    ]
    freq = [0.0, 10.0, 20.0, 50.0, 100.0, 500.0, 3000.0, 6750.0,
            7000.0, 8000.0, 9000.0]
    for vl, vh, lo, hi, wp in cases:
        r = solve("vtln_warp", vtln_low=vl, vtln_high=vh,
                  low_freq=lo, high_freq=hi, vtln_warp=wp, freq=freq)
        o = _o_vtln(vl, vh, lo, hi, wp, freq)
        for k in o:
            _must(_eq(r[k], o[k]), "vtln_warp %s mismatch (warp=%s)" % (k, wp))
        # 双向：outside_low_high 最后覆盖
        for i, f in enumerate(freq):
            if f < lo or f > hi:
                _must(_eq(r["res"][i], f),
                      "vtln outside freq must be preserved verbatim")
        # 双向：三段线性中段是 scale*f，不是恒等
        if wp != 1.0:
            mid = [r["res"][i] for i, f in enumerate(freq)
                   if lo <= f < min(vl * max(1.0, wp), vh * min(1.0, wp))]
            if mid:
                _must(any(abs(v - f) > 1e-6 for v, f in
                          zip(mid, [f for f in freq
                                    if lo <= f < min(vl * max(1.0, wp),
                                                     vh * min(1.0, wp))])),
                      "vtln center segment must apply scale=1/warp")
    # 双向：warp=2.0 时 l>h，重叠区必须用 ③ before_l 而非 ① after_h
    vl, vh, lo, hi, wp = 7000.0, 100.0, 20.0, 8000.0, 2.0
    r = solve("vtln_warp", vtln_low=vl, vtln_high=vh, low_freq=lo,
              high_freq=hi, vtln_warp=wp, freq=[500.0])
    o = _o_vtln(vl, vh, lo, hi, wp, [500.0])
    _must(_eq(r["res"], o["res"]), "vtln overlap must follow before_l (③)")
    wrong_after_h = [hi + o["scale_right"] * (500.0 - hi)]
    _must_differ(r["res"], wrong_after_h,
                 "vtln overlap must NOT be after_h (①) value")
    wrong_identity = [500.0]
    _must_differ(r["res"], wrong_identity,
                 "vtln overlap must NOT be identity")

    # 域约束
    _raises(lambda: solve("vtln_warp", vtln_low=10.0, vtln_high=7500.0,
                          low_freq=20.0, high_freq=8000.0,
                          vtln_warp=0.9, freq=[1.0]), "vtln_low",
            "vtln_low <= low_freq")
    _raises(lambda: solve("vtln_warp", vtln_low=100.0, vtln_high=9000.0,
                          low_freq=20.0, high_freq=8000.0,
                          vtln_warp=0.9, freq=[1.0]), "vtln_high",
            "vtln_high >= high_freq")
    _raises(lambda: solve("vtln_warp", vtln_low=100.0, vtln_high=7500.0,
                          low_freq=20.0, high_freq=8000.0,
                          vtln_warp=0.0, freq=[1.0]), "warp",
            "vtln_warp <= 0")


# =========================================================================
# check 2：get_mel_banks
# =========================================================================
def check_mel_banks(solve):
    cases = [
        (5, 32, 20.0, 0.0, 100.0, -500.0, 1.0),
        (5, 32, 20.0, 0.0, 100.0, -500.0, 0.9),
        (5, 32, 20.0, 0.0, 100.0, -500.0, 1.1),
        (4, 16, 0.0, -1000.0, 100.0, -500.0, 1.0),
        (7, 64, 50.0, 7000.0, 200.0, 6000.0, 1.0),
        (7, 64, 50.0, 7000.0, 200.0, 6000.0, 0.85),
        (8, 32, 20.0, 0.0, 100.0, -500.0, 1.0),
    ]
    for nb, pad, lo, hi, vl, vh, wp in cases:
        r = solve("mel_banks", num_bins=nb, window_length_padded=pad,
                  sample_freq=SF, low_freq=lo, high_freq=hi,
                  vtln_low=vl, vtln_high=vh, vtln_warp=wp)
        o = _o_banks(nb, pad, SF, lo, hi, vl, vh, wp)
        for k in ("bins", "center_freqs", "mel_low_freq", "mel_high_freq",
                  "mel_freq_delta", "num_fft_bins", "nyquist",
                  "fft_bin_width", "high_freq_eff", "vtln_high_eff"):
            _must(_eq(r[k], o[k]), "mel_banks %s mismatch" % k)
        # 双向：delta 分母是 num_bins+1
        alt = (o["mel_high_freq"] - o["mel_low_freq"]) / nb
        _must(not _eq(r["mel_freq_delta"], alt),
              "mel_freq_delta must divide by num_bins+1, not num_bins")
        # 双向：center_freqs 形状 (n,1) 而非 (n,)
        _must(len(r["center_freqs"]) == nb
              and all(len(row) == 1 for row in r["center_freqs"]),
              "center_freqs must be (num_bins, 1)")
        # 双向：bins 非负且至少有一行非零
        _must(all(v >= -1e-12 for row in r["bins"] for v in row),
              "bins must be clamped non-negative")
        _must(any(v > 0 for row in r["bins"] for v in row),
              "bins must have non-zero triangular support")
        # high_freq <= 0 时加 nyquist
        if hi <= 0:
            _must(_eq(r["high_freq_eff"], SF / 2.0 + hi),
                  "high_freq<=0 must be offset by nyquist")
        if vh < 0:
            _must(_eq(r["vtln_high_eff"], SF / 2.0 + vh),
                  "vtln_high<0 must be offset by nyquist")

    # 双向：warp!=1 走 mask 分支，与 warp=1 的 min/max 分支结果不同
    a = solve("mel_banks", num_bins=5, window_length_padded=32,
              sample_freq=SF, low_freq=20.0, high_freq=0.0,
              vtln_low=100.0, vtln_high=-500.0, vtln_warp=1.0)
    b = solve("mel_banks", num_bins=5, window_length_padded=32,
              sample_freq=SF, low_freq=20.0, high_freq=0.0,
              vtln_low=100.0, vtln_high=-500.0, vtln_warp=0.9)
    _must_differ(a["bins"], b["bins"],
                 "warp!=1 must use masked branches, not min/max")
    _must_differ(a["center_freqs"], b["center_freqs"],
                 "warp!=1 must warp center freqs")

    # 域约束
    _raises(lambda: solve("mel_banks", num_bins=3,
                          window_length_padded=32, sample_freq=SF,
                          low_freq=20.0, high_freq=0.0, vtln_low=100.0,
                          vtln_high=-500.0, vtln_warp=1.0),
            "at least 3", "num_bins <= 3")
    _raises(lambda: solve("mel_banks", num_bins=5,
                          window_length_padded=31, sample_freq=SF,
                          low_freq=20.0, high_freq=0.0, vtln_low=100.0,
                          vtln_high=-500.0, vtln_warp=1.0),
            "even", "odd padded window size")
    _raises(lambda: solve("mel_banks", num_bins=5,
                          window_length_padded=32, sample_freq=SF,
                          low_freq=9000.0, high_freq=0.0, vtln_low=100.0,
                          vtln_high=-500.0, vtln_warp=1.0),
            "Bad values", "low_freq above nyquist")
    _raises(lambda: solve("mel_banks", num_bins=5,
                          window_length_padded=32, sample_freq=SF,
                          low_freq=20.0, high_freq=0.0, vtln_low=9000.0,
                          vtln_high=-500.0, vtln_warp=0.9),
            "vtln-low", "invalid vtln cutoffs")


# =========================================================================
# check 3：framing（分帧 / 窗函数 / 前处理链）
# =========================================================================
def check_framing(solve):
    w = _wave(90, 77)
    cases = [
        _fr(),
        _fr(window_type="hanning"),
        _fr(window_type="povey"),
        _fr(window_type="rectangular"),
        _fr(window_type="blackman"),
        _fr(window_type="blackman", blackman_coeff=0.5),
        _fr(snip_edges=False),
        _fr(snip_edges=False, window_type="povey"),
        _fr(raw_energy=False),
        _fr(remove_dc_offset=False),
        _fr(preemphasis_coefficient=0.0),
        _fr(energy_floor=0.0),
        _fr(energy_floor=10.0),
        _fr(window_size=24, padded_window_size=32),
        _fr(window_shift=8),
        _fr(window_size=16, window_shift=20, snip_edges=False,
            padded_window_size=32),  # pad < 0 分支
    ]
    for fr in cases:
        r = solve("framing", waveform=w, **fr)
        o = _o_chain(w, fr)
        for k in ("raw_frames", "strided", "window_function",
                  "signal_log_energy", "num_frames", "epsilon",
                  "padded_window_size"):
            _must(_eq(r[k], o[k]), "framing %s mismatch (wt=%s snip=%s)"
                  % (k, fr["window_type"], fr["snip_edges"]))
        # 双向：padded 右列零填充
        if fr["padded_window_size"] > fr["window_size"]:
            for row in r["strided"]:
                _must(all(abs(v) < 1e-12 for v in
                          row[fr["window_size"]:]),
                      "padded columns must be zero")

    # 双向：povey = hann^0.85，既不是 hann 也不是 hamming
    hann = solve("framing", waveform=w, **_fr(window_type="hanning"))
    povey = solve("framing", waveform=w, **_fr(window_type="povey"))
    hamming = solve("framing", waveform=w, **_fr(window_type="hamming"))
    _must_differ(povey["window_function"], hann["window_function"],
                 "povey must be hann^0.85, not plain hann")
    _must_differ(povey["window_function"], hamming["window_function"],
                 "povey must not be hamming")
    # 双向：blackman 手写系数随 blackman_coeff 变化
    b042 = solve("framing", waveform=w, **_fr(window_type="blackman"))
    b050 = solve("framing", waveform=w,
                 **_fr(window_type="blackman", blackman_coeff=0.5))
    _must_differ(b042["window_function"], b050["window_function"],
                 "blackman must use blackman_coeff")

    # 双向：snip_edges=False 反射填充 ≠ 零填充
    sn = solve("framing", waveform=w, **_fr(snip_edges=False))
    zero_pad = [list(row) for row in
                solve("framing", waveform=w, **_fr(snip_edges=True))["raw_frames"]]
    m = len(sn["raw_frames"])
    zero = []
    for i in range(m):
        start = i * 16
        zero.append([w[start + j] if start + j < len(w) else 0.0
                     for j in range(32)])
    _must_differ(sn["raw_frames"], zero,
                 "snip_edges=False must reflect-pad, not zero-pad")

    # 双向：preemphasis 用 replicate pad ⇒ 首样本 (1-c)*x0
    r = solve("framing", waveform=w,
              **_fr(window_type="rectangular", remove_dc_offset=False,
                    preemphasis_coefficient=0.5))
    f0 = r["raw_frames"][0][0]
    _must(_eq(r["strided"][0][0], 0.5 * f0),
          "preemphasis first sample must be (1-c)*x0 via replicate pad")
    _must_differ(r["strided"][0][0], f0,
                 "preemphasis must NOT leave first sample unchanged")

    # 双向：raw_energy 顺序
    raw = solve("framing", waveform=w, **_fr(raw_energy=True))
    post = solve("framing", waveform=w, **_fr(raw_energy=False))
    _must_differ(raw["signal_log_energy"], post["signal_log_energy"],
                 "raw_energy must be computed BEFORE preemphasis/window")

    # 双向：energy_floor 在 log 域取 max
    z = [0.0] * 90
    for floor in (0.0, 1.0, 10.0):
        rr = solve("framing", waveform=z,
                   **_fr(energy_floor=floor, remove_dc_offset=False,
                         preemphasis_coefficient=0.0,
                         window_type="rectangular"))
        expect = max(math.log(EPS), math.log(floor)) if floor else math.log(EPS)
        _must(_eq(rr["signal_log_energy"][0], expect),
              "log-energy floor must be applied in log domain")

    # dither：noise 必须提供，且乘 dither 后叠加
    fr = _fr(dither=0.5)
    _raises(lambda: solve("framing", waveform=w, **fr),
            "dither noise", "dither without noise")
    nz = _noise(len(solve("framing", waveform=w, **_fr())["raw_frames"]),
                32, 555)
    fr2 = _fr(dither=0.5, noise=nz)
    r2 = solve("framing", waveform=w, **fr2)
    o2 = _o_chain(w, fr2)
    _must(_eq(r2["strided"], o2["strided"]), "dither chain mismatch")
    _must_differ(r2["strided"],
                 solve("framing", waveform=w, **_fr())["strided"],
                 "dither must actually perturb the frames")

    # 域约束
    _raises(lambda: solve("framing", waveform=w,
                          **_fr(window_type="bogus")),
            "Invalid window type", "unknown window type")
    _raises(lambda: solve("framing", waveform=w, **_fr(window_shift=0)),
            "window_shift", "non-positive window_shift")
    _raises(lambda: solve("framing", waveform=w,
                          **_fr(padded_window_size=16)),
            "padded_window_size", "padded < window_size")
    _raises(lambda: solve("framing", waveform=w,
                          **_fr(padded_window_size=33)),
            "even", "odd padded size")


# =========================================================================
# check 4：fbank
# =========================================================================
def check_fbank(solve):
    cases = [
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, True, True, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, False, True, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, True, False, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, True, True, True, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, True, True, True, True, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, True, True, False, False, True),
        (5, 20.0, 0.0, 100.0, -500.0, 0.9, True, True, True, False, False),
        (6, 20.0, 0.0, 100.0, -500.0, 1.1, True, True, False, False, False),
        (7, 50.0, 7000.0, 200.0, 6000.0, 1.0, True, True, False, False, False),
    ]
    for ci, (nb, lo, hi, vl, vh, wp, up, ul, ue, htk, sm) in enumerate(cases):
        w = _wave(100, 300 + ci)
        fr = _fr()
        r = solve("fbank", waveform=w, sample_freq=SF, num_mel_bins=nb,
                  low_freq=lo, high_freq=hi, vtln_low=vl, vtln_high=vh,
                  vtln_warp=wp, use_power=up, use_log_fbank=ul,
                  use_energy=ue, htk_compat=htk, subtract_mean=sm, **fr)
        o = _o_fbank(w, fr, SF, nb, lo, hi, vl, vh, wp, up, ul, ue, htk, sm)
        for k in ("features", "bins", "bins_padded", "spectrum",
                  "mel_energies", "signal_log_energy", "center_freqs",
                  "raw_frames", "strided", "window_function"):
            _must(_eq(r[k], o[k]), "fbank %s mismatch (case %d)" % (k, ci))
        # 双向：bins_padded 最后一列全 0
        for row in r["bins_padded"]:
            _must(abs(row[-1]) < 1e-12,
                  "bins_padded last column must be zero")
        # 双向：subtract_mean 后列均值为 0
        if sm:
            ncol = len(r["features"][0])
            for j in range(ncol):
                mu = sum(row[j] for row in r["features"]) / len(r["features"])
                _must(abs(mu) < 1e-9, "CMS column mean must be zero")

    w = _wave(100, 401)
    fr = _fr()
    base = dict(waveform=w, sample_freq=SF, num_mel_bins=5, low_freq=20.0,
                high_freq=0.0, vtln_low=100.0, vtln_high=-500.0,
                vtln_warp=1.0, subtract_mean=False, **fr)
    # 双向：use_power
    p1 = solve("fbank", use_power=True, use_log_fbank=True,
               use_energy=False, htk_compat=False, **base)
    p0 = solve("fbank", use_power=False, use_log_fbank=True,
               use_energy=False, htk_compat=False, **base)
    _must_differ(p1["spectrum"], p0["spectrum"],
                 "use_power must square the magnitude spectrum")
    # 双向：use_log_fbank
    l1 = solve("fbank", use_power=True, use_log_fbank=True,
               use_energy=False, htk_compat=False, **base)
    l0 = solve("fbank", use_power=True, use_log_fbank=False,
               use_energy=False, htk_compat=False, **base)
    _must_differ(l1["features"], l0["features"],
                 "use_log_fbank must apply log(max(x, eps))")
    # 双向：use_energy 位置由 htk_compat 决定
    e_first = solve("fbank", use_power=True, use_log_fbank=True,
                    use_energy=True, htk_compat=False, **base)
    e_last = solve("fbank", use_power=True, use_log_fbank=True,
                   use_energy=True, htk_compat=True, **base)
    _must_differ(e_first["features"], e_last["features"],
                 "htk_compat must move energy column to the end")
    _must(_eq([row[0] for row in e_first["features"]],
              e_first["signal_log_energy"]),
          "energy must be the FIRST column when htk_compat=False")
    _must(_eq([row[-1] for row in e_last["features"]],
              e_last["signal_log_energy"]),
          "energy must be the LAST column when htk_compat=True")
    # 双向：subtract_mean
    c0 = solve("fbank", use_power=True, use_log_fbank=True,
               use_energy=False, htk_compat=False, subtract_mean=False,
               **{k: v for k, v in base.items() if k != "subtract_mean"})
    c1 = solve("fbank", use_power=True, use_log_fbank=True,
               use_energy=False, htk_compat=False, subtract_mean=True,
               **{k: v for k, v in base.items() if k != "subtract_mean"})
    _must_differ(c0["features"], c1["features"],
                 "subtract_mean must subtract per-column means")


# =========================================================================
# check 5：mfcc
# =========================================================================
def check_mfcc(solve):
    cases = [
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 22.0, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 3, 22.0, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 0.0, False, False, False),
        (6, 20.0, 0.0, 100.0, -500.0, 1.0, 4, 30.0, False, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 22.0, True, False, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 22.0, False, True, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 22.0, True, True, False),
        (5, 20.0, 0.0, 100.0, -500.0, 1.0, 5, 22.0, True, True, True),
    ]
    for ci, (nb, lo, hi, vl, vh, wp, nc, lif, ue, htk, sm) in enumerate(cases):
        w = _wave(100, 500 + ci)
        fr = _fr()
        r = solve("mfcc", waveform=w, sample_freq=SF, num_mel_bins=nb,
                  low_freq=lo, high_freq=hi, vtln_low=vl, vtln_high=vh,
                  vtln_warp=wp, num_ceps=nc, cepstral_lifter=lif,
                  use_energy=ue, htk_compat=htk, subtract_mean=sm, **fr)
        o = _o_mfcc(w, fr, SF, nb, lo, hi, vl, vh, wp, nc, lif, ue, htk, sm)
        for k in ("features", "dct_matrix", "lifter_coeffs",
                  "fbank_features"):
            _must(_eq(r[k], o[k]), "mfcc %s mismatch (case %d)" % (k, ci))
        # 双向：dct 首列等权 sqrt(1/num_mel_bins)
        _must(all(_eq(row[0], math.sqrt(1.0 / nb)) for row in r["dct_matrix"]),
              "dct first column must be sqrt(1/num_mel_bins)")
        _must(len(r["dct_matrix"][0]) == nc and len(r["dct_matrix"]) == nb,
              "dct_matrix shape must be (num_mel_bins, num_ceps)")
        # 双向：lifter C0 恒为 1
        _must(_eq(r["lifter_coeffs"][0], 1.0),
              "lifter C0 must be exactly 1.0")
        if lif != 0.0:
            _must(any(not _eq(v, 1.0) for v in r["lifter_coeffs"][1:]),
                  "lifter must scale coefficients above C0")
        else:
            _must(_eq(r["lifter_coeffs"], [1.0] * nc),
                  "lifter=0 must disable all scaling")
        # 双向：subtract_mean
        if sm:
            for j in range(len(r["features"][0])):
                mu = sum(row[j] for row in r["features"]) / len(r["features"])
                _must(abs(mu) < 1e-9, "CMS column mean must be zero")

    w = _wave(100, 601)
    fr = _fr()
    base = dict(waveform=w, sample_freq=SF, num_mel_bins=5, low_freq=20.0,
                high_freq=0.0, vtln_low=100.0, vtln_high=-500.0,
                vtln_warp=1.0, num_ceps=5, cepstral_lifter=22.0,
                subtract_mean=False, **fr)
    no_htk = solve("mfcc", use_energy=False, htk_compat=False, **base)
    htk = solve("mfcc", use_energy=False, htk_compat=True, **base)
    _must_differ(no_htk["features"], htk["features"],
                 "htk_compat must move C0 to the last column")
    # 双向：htk 且 !use_energy 时末尾 C0 乘 sqrt(2)
    _must(_eq([row[-1] for row in htk["features"]],
              [row[0] * math.sqrt(2.0) for row in no_htk["features"]]),
          "htk must scale C0 by sqrt(2) when use_energy=False")
    # 双向：use_energy 时首列被 signal_log_energy 替换
    ue = solve("mfcc", use_energy=True, htk_compat=False, **base)
    fb = solve("fbank", waveform=w, sample_freq=SF, num_mel_bins=5,
               low_freq=20.0, high_freq=0.0, vtln_low=100.0,
               vtln_high=-500.0, vtln_warp=1.0, use_power=True,
               use_log_fbank=True, use_energy=True, htk_compat=False,
               subtract_mean=False, **fr)
    _must(_eq([row[0] for row in ue["features"]],
              [row[0] for row in fb["features"]]),
          "use_energy must replace C0 with signal_log_energy")
    _must_differ(ue["features"], no_htk["features"],
                 "use_energy must change C0")
    # 双向：lifter 开关
    l0 = solve("mfcc", use_energy=False, htk_compat=False,
               **{**base, "cepstral_lifter": 0.0})
    l1 = solve("mfcc", use_energy=False, htk_compat=False, **base)
    _must_differ(l0["features"], l1["features"],
                 "cepstral_lifter must actually scale cepstra")

    # 域约束
    _raises(lambda: solve("mfcc", waveform=w, sample_freq=SF,
                          num_mel_bins=5, low_freq=20.0, high_freq=0.0,
                          vtln_low=100.0, vtln_high=-500.0, vtln_warp=1.0,
                          num_ceps=6, cepstral_lifter=22.0,
                          use_energy=False, htk_compat=False,
                          subtract_mean=False, **fr),
            "num_ceps", "num_ceps > num_mel_bins")


# =========================================================================
# check 6：_FRAMING_DEFAULTS 默认值（povey / 0.97 / 1.0 / 0.42 / True）
# =========================================================================
def check_defaults(solve):
    w = _wave(90, 88)
    r = solve("framing", waveform=w, window_size=32, window_shift=16,
              padded_window_size=32)
    o = _o_chain(w, _fr(window_type="povey"))
    for k in ("raw_frames", "strided", "window_function",
              "signal_log_energy", "num_frames", "epsilon",
              "padded_window_size"):
        _must(_eq(r[k], o[k]), "framing default %s mismatch" % k)
    _must(_eq(r["window_function"], _o_window("povey", 32, 0.42)),
          "default window_type must be povey")
    # 全零波形：默认 energy_floor=1.0 必须把 log-energy clamp 到 log(1)=0
    z = [0.0] * 90
    rz = solve("framing", waveform=z, window_size=32, window_shift=16,
               padded_window_size=32)
    _must(_eq(rz["signal_log_energy"][0], 0.0),
          "default energy_floor must clamp log-energy to log(1.0)=0")
    # 默认 blackman_coeff=0.42（povey 窗不依赖它，须显式走 blackman 分支）
    rb = solve("framing", waveform=w, window_size=32, window_shift=16,
               padded_window_size=32, window_type="blackman")
    _must(_eq(rb["window_function"], _o_window("blackman", 32, 0.42)),
          "default blackman_coeff must be 0.42")


# =========================================================================
# check 7：solve 分派
# =========================================================================
def check_unknown_mode(solve):
    _raises(lambda: solve("nope"), "unknown mode", "unknown mode")
