#!/usr/bin/env python3
"""857 reference solution: torchaudio functional waveform-operations ledger.

Pure Python, no torch, no numpy, no global RNG.  Six independent modes
reproduce the official ``torchaudio.functional`` mechanisms:

    mu_law       : mu_law_encoding / mu_law_decoding
    convolve     : direct conv1d method (swap, broadcast, mode slicing)
    fftconvolve  : rfft/irfft product (DFT) + mode slicing
    slice_mode   : _apply_convolve_mode center cropping
    add_noise    : SNR scaling with optional lengths mask
    phase_vocoder: phase accumulation and polar reconstruction
"""

import cmath
import math


# ---------------------------------------------------------------------------
# nested-list utilities
# ---------------------------------------------------------------------------
def shape_of(x):
    if not isinstance(x, list):
        return []
    if not x:
        return [0]
    return [len(x)] + shape_of(x[0])


def numel(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def flatten(x):
    if not isinstance(x, list):
        return [x]
    out = []
    for v in x:
        out.extend(flatten(v))
    return out


def unflatten(vals, shape):
    if not shape:
        return vals[0]
    step = numel(shape[1:])
    return [unflatten(vals[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def reshape(x, shape):
    vals = flatten(x)
    if numel(shape) != len(vals):
        raise ValueError("cannot reshape %d elements into %r" % (len(vals), shape))
    return unflatten(vals, shape)


def broadcast_shapes(a, b):
    ra, rb = list(reversed(a)), list(reversed(b))
    out = []
    for i in range(max(len(a), len(b))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError("shapes are not broadcastable: %r and %r" % (a, b))
        out.append(max(da, db))
    return list(reversed(out))


def broadcast_to(x, shape):
    s = shape_of(x)
    if s == shape:
        return x
    if not s:
        return unflatten([x] * numel(shape), shape)
    if len(shape) < len(s):
        raise ValueError("cannot broadcast %r to %r" % (s, shape))
    off = len(shape) - len(s)

    def build(rest, prefix):
        if not rest:
            v = x
            for m, d in enumerate(s):
                v = v[0 if d == 1 else prefix[off + m]]
            return v
        return [build(rest[1:], prefix + [i]) for i in range(rest[0])]

    return build(shape, [])


def slice_last(x, start, end):
    s = shape_of(x)
    if len(s) <= 1:
        return x[start:end]
    return [slice_last(v, start, end) for v in x]


def map_vals(x, fn):
    if not isinstance(x, list):
        return fn(x)
    return [map_vals(v, fn) for v in x]


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def require_numbers(x, what):
    vals = flatten(x)
    if not vals or not all(is_number(v) for v in vals):
        raise ValueError("%s must be a non-empty numeric array" % what)
    if not isinstance(x, list):
        raise ValueError("%s must be an array" % what)


def require_int(v, what, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("%s must be an integer" % what)
    if minimum is not None and v < minimum:
        raise ValueError("%s must be at least %d" % (what, minimum))


# ---------------------------------------------------------------------------
# mode 1: mu_law
# ---------------------------------------------------------------------------
def _mu_law(kw):
    x = kw["x"]
    require_numbers(x, "x")
    qc = kw["quantization_channels"]
    require_int(qc, "quantization_channels", 2)
    mu = float(qc - 1)
    log1p_mu = math.log1p(mu)

    def enc(v):
        if v < -1.0 or v > 1.0:
            raise ValueError("x must be scaled to between -1 and 1")
        x_mu = math.copysign(math.log1p(mu * abs(v)) / log1p_mu, v)
        # official .to(torch.int64) truncates toward zero; the +0.5 makes it
        # a round-half-up on the non-negative encoded value
        return int((x_mu + 1) / 2 * mu + 0.5)

    def dec(e):
        if e < 0 or e > qc - 1:
            raise ValueError("x_mu must be between 0 and quantization_channels-1")
        dx = (float(e) / mu) * 2 - 1.0
        return math.copysign(
            (math.exp(abs(dx) * log1p_mu) - 1.0) / mu, dx)

    encoded = map_vals(x, enc)
    decoded = map_vals(encoded, dec)
    return {
        "encoded": encoded,
        "decoded": decoded,
        "mu": mu,
        "quantization_channels": qc,
        "shape": shape_of(encoded),
    }


# ---------------------------------------------------------------------------
# convolution helpers
# ---------------------------------------------------------------------------
def _check_shape_compatible(x, y):
    sx, sy = shape_of(x), shape_of(y)
    if len(sx) != len(sy):
        raise ValueError("The operands must be the same dimension "
                         "(got %d and %d)." % (len(sx), len(sy)))
    for i in range(len(sx) - 1):
        a, b = sx[i], sy[i]
        if not (a == b or a == 1 or b == 1):
            raise ValueError("Leading dimensions of x and y are not "
                             "broadcastable (got %r and %r)." % (sx, sy))


def _check_mode(mode):
    if mode not in ("full", "valid", "same"):
        raise ValueError("Unrecognized mode value %r. Please specify one of "
                         "['full', 'valid', 'same']." % (mode,))


def _slice_mode(full, x_length, y_length, mode):
    _check_mode(mode)
    if mode == "full":
        return full
    total = shape_of(full)[-1]
    if mode == "valid":
        target = max(x_length, y_length) - min(x_length, y_length) + 1
        start = (total - target) // 2
    else:
        target = x_length
        start = (total - x_length) // 2
    return slice_last(full, start, start + target)


def _rows(x, lead):
    """Flatten leading dims into [B, N] with broadcasting."""
    s = shape_of(x)
    b = broadcast_to(x, lead + [s[-1]])
    return reshape(b, [numel(lead), s[-1]])


def _conv_full(a, b):
    n, m = len(a), len(b)
    return [sum(a[i] * b[k - i] for i in range(n) if 0 <= k - i < m)
            for k in range(n + m - 1)]


def _dft(vals, n):
    out = []
    for k in range(n):
        acc = 0j
        for j, v in enumerate(vals):
            acc += v * cmath.exp(-2j * math.pi * j * k / n)
        out.append(acc)
    return out


def _idft(vals, n):
    out = []
    for j in range(n):
        acc = 0j
        for k, v in enumerate(vals):
            acc += v * cmath.exp(2j * math.pi * j * k / n)
        out.append(acc.real / n)
    return out


def _convolve(kw, use_fft=False):
    x, y = kw["x"], kw["y"]
    require_numbers(x, "x")
    require_numbers(y, "y")
    # external name is conv_mode: `mode` is taken by solve()'s dispatch
    mode = kw.get("conv_mode", "full")
    _check_shape_compatible(x, y)
    _check_mode(mode)
    sx, sy = shape_of(x), shape_of(y)
    x_size, y_size = sx[-1], sy[-1]
    if x_size == 0 or y_size == 0:
        raise ValueError("x and y must be non-empty along the last dimension")
    swapped = False
    if not use_fft and x_size < y_size:
        x, y = y, x
        swapped = True
    sx2, sy2 = shape_of(x), shape_of(y)
    lead = broadcast_shapes(sx2[:-1] if sx2[:-1] else [1],
                            sy2[:-1] if sy2[:-1] else [1])
    if sx2[:-1] == [] and sy2[:-1] == []:
        lead = []
    xr, yr = _rows(x, lead), _rows(y, lead)
    if use_fft:
        n = shape_of(x)[-1] + shape_of(y)[-1] - 1
        rows = []
        for a, b in zip(xr, yr):
            pa = list(a) + [0.0] * (n - len(a))
            pb = list(b) + [0.0] * (n - len(b))
            prod = [p * q for p, q in zip(_dft(pa, n), _dft(pb, n))]
            rows.append(_idft(prod, n))
    else:
        rows = [_conv_full(a, b) for a, b in zip(xr, yr)]
    L = len(rows[0])
    full = unflatten(flatten(rows), lead + [L])
    out = _slice_mode(full, x_size, y_size, mode)
    return {
        "out": out,
        "shape": shape_of(out),
        "full": full,
        "full_length": L,
        "mode": mode,
        "swapped": swapped,
        "x_length": x_size,
        "y_length": y_size,
        "leading_shape": lead,
        "num_signals": max(1, numel(lead)),
    }


def _slice_mode_only(kw):
    full = kw["full"]
    require_numbers(full, "full")
    x_length = kw["x_length"]
    y_length = kw["y_length"]
    require_int(x_length, "x_length", 1)
    require_int(y_length, "y_length", 1)
    mode = kw.get("conv_mode", "full")
    out = _slice_mode(full, x_length, y_length, mode)
    total = shape_of(full)[-1]
    if mode == "full":
        target, start = total, 0
    elif mode == "valid":
        target = max(x_length, y_length) - min(x_length, y_length) + 1
        start = (total - target) // 2
    else:
        target, start = x_length, (total - x_length) // 2
    return {"out": out, "shape": shape_of(out), "mode": mode,
            "start_idx": start, "target_length": target}


# ---------------------------------------------------------------------------
# mode 4: add_noise
# ---------------------------------------------------------------------------
def _add_noise(kw):
    wf, noise = kw["waveform"], kw["noise"]
    require_numbers(wf, "waveform")
    require_numbers(noise, "noise")
    snr = kw["snr"]
    lengths = kw.get("lengths")
    sw, sn, ss = shape_of(wf), shape_of(noise), shape_of(snr)
    if lengths is not None:
        sl = shape_of(lengths)
    else:
        sl = ss
    if not (len(sw) - 1 == len(sn) - 1 == len(ss) and len(sl) == len(ss)):
        raise ValueError("Input leading dimensions don't match.")
    L = sw[-1]
    if L != sn[-1]:
        raise ValueError("Length dimensions of waveform and noise don't "
                         "match (got %d and %d)." % (L, sn[-1]))

    wlead, nlead = sw[:-1], sn[:-1]
    lead = broadcast_shapes(wlead, broadcast_shapes(nlead,
                            broadcast_shapes(ss, sl)))
    B = max(1, numel(lead))
    wb = reshape(broadcast_to(wf, lead + [L]), [B, L])
    nb = reshape(broadcast_to(noise, lead + [L]), [B, L])
    sb = reshape(broadcast_to(snr, lead) if lead else [snr], [B]) \
        if lead else [snr] * B
    if lengths is None:
        lb = [L] * B
    else:
        lb = reshape(broadcast_to(lengths, lead) if lead else [lengths], [B])

    out, scales, es_list, en_list = [], [], [], []
    for b in range(B):
        es = sum(wb[b][i] ** 2 for i in range(L) if i < lb[b])
        en = sum(nb[b][i] ** 2 for i in range(L) if i < lb[b])
        if es <= 0 or en <= 0:
            raise ValueError("energy_signal and energy_noise must be positive")
        original = 10 * (math.log10(es) - math.log10(en))
        scale = 10 ** ((original - sb[b]) / 20.0)
        scales.append(scale)
        es_list.append(es)
        en_list.append(en)
        # official adds scaled *unmasked* noise to the *unmasked* waveform
        out.append([wb[b][i] + scale * nb[b][i] for i in range(L)])
    out_shape = lead + [L]
    return {
        "out": unflatten(flatten(out), out_shape),
        "shape": out_shape,
        "scale": unflatten(scales, lead) if lead else scales[0],
        "energy_signal": unflatten(es_list, lead) if lead else es_list[0],
        "energy_noise": unflatten(en_list, lead) if lead else en_list[0],
        "leading_shape": lead,
        "length": L,
    }


# ---------------------------------------------------------------------------
# mode 5: phase_vocoder
# ---------------------------------------------------------------------------
def _phase_vocoder(kw):
    real, imag = kw["real"], kw["imag"]
    require_numbers(real, "real")
    require_numbers(imag, "imag")
    sr, si = shape_of(real), shape_of(imag)
    if sr != si:
        raise ValueError("real and imag must have the same shape")
    if len(sr) < 2:
        raise ValueError("complex_specgrams must have (..., freq, frames) shape")
    rate = kw["rate"]
    if not is_number(rate) or rate <= 0:
        raise ValueError("rate must be a positive number")
    pa = kw["phase_advance"]
    require_numbers(pa, "phase_advance")
    lead, F, T = sr[:-2], sr[-2], sr[-1]
    if T == 0 or F == 0:
        raise ValueError("complex_specgrams must be non-empty")
    # phase_advance is (freq, 1) in the official API; reduce to per-bin scalars
    pav = flatten(pa)
    if len(pav) != F:
        raise ValueError("phase_advance must have one value per freq bin")
    B = max(1, numel(lead))
    rv = reshape(real, [B, F, T])
    iv = reshape(imag, [B, F, T])
    spec = [[[complex(rv[b][f][t], iv[b][f][t]) for t in range(T)]
             for f in range(F)] for b in range(B)]

    if rate == 1.0:
        n = T
        out_real, out_imag = real, imag
        ts = [float(t) for t in range(T)]
    else:
        # torch.arange(0, T, rate): ceil(T/rate) samples, t_j = j*rate
        n = int(math.ceil(T / rate - 1e-12))
        ts = [j * rate for j in range(n)]
        alphas = [t % 1.0 for t in ts]
        phase_0 = [[cmath.phase(spec[b][f][0]) for f in range(F)]
                   for b in range(B)]
        padded = [[row + [0j, 0j] for row in spec[b]] for b in range(B)]
        out_real, out_imag = [], []
        for b in range(B):
            rr, ii = [], []
            for f in range(F):
                phases, norms0, norms1 = [], [], []
                for t in ts:
                    i0, i1 = int(t), int(t + 1)
                    s0, s1 = padded[b][f][i0], padded[b][f][i1]
                    a0, a1 = cmath.phase(s0), cmath.phase(s1)
                    norms0.append(abs(s0))
                    norms1.append(abs(s1))
                    ph = a1 - a0 - pav[f]
                    ph = ph - 2 * math.pi * round(ph / (2 * math.pi))
                    phases.append(ph + pav[f])
                shifted = [phase_0[b][f]] + phases[:-1]
                acc, running = [], 0.0
                for v in shifted:
                    running += v
                    acc.append(running)
                rr.append([(alphas[j] * norms1[j] +
                            (1 - alphas[j]) * norms0[j]) * math.cos(acc[j])
                           for j in range(n)])
                ii.append([(alphas[j] * norms1[j] +
                            (1 - alphas[j]) * norms0[j]) * math.sin(acc[j])
                           for j in range(n)])
            out_real.append(rr)
            out_imag.append(ii)
        out_real = unflatten(flatten(out_real), lead + [F, n])
        out_imag = unflatten(flatten(out_imag), lead + [F, n])
    return {
        "real": out_real,
        "imag": out_imag,
        "shape": lead + [F, n],
        "time_steps": ts,
        "alphas": [t % 1.0 for t in ts],
        "num_frames_in": T,
        "num_frames_out": n,
        "freq": F,
        "rate": rate,
    }


# ---------------------------------------------------------------------------
MODES = ("mu_law", "convolve", "fftconvolve", "slice_mode",
         "add_noise", "phase_vocoder")


def solve(mode, **kw):
    if mode == "mu_law":
        return _mu_law(kw)
    if mode == "convolve":
        return _convolve(kw, use_fft=False)
    if mode == "fftconvolve":
        return _convolve(kw, use_fft=True)
    if mode == "slice_mode":
        return _slice_mode_only(kw)
    if mode == "add_noise":
        return _add_noise(kw)
    if mode == "phase_vocoder":
        return _phase_vocoder(kw)
    raise ValueError("unknown mode: %r" % (mode,))
