import math

import torch

# 两种 FP6 子格式：(指数位, 尾数位, bias)。均无 inf / NaN，最大码点即饱和点。
SPECS = {
    "e3m2": (3, 2, 3),  # 动态范围大：max = 1.75 × 2^4 = 28
    "e2m3": (2, 3, 1),  # 精度高：max = 1.875 × 2^2 = 7.5
}
LIMITS = {
    name: (1.0 + (2 ** m - 1) / float(2 ** m)) * float(2 ** ((2 ** e - 1) - b))
    for name, (e, m, b) in SPECS.items()
}


def _table(name):
    """按位域公式建 64 个码点；E3M2 的 0.25 有两个码，两个都留在表里。"""
    ebits, mbits, bias = SPECS[name]
    rows = []
    for code in range(64):
        neg = bool(code & 32)
        e = (code >> mbits) & (2 ** ebits - 1)
        m = code & (2 ** mbits - 1)
        if e == 0:
            mag = m / float(2 ** mbits) * float(2 ** (1 - bias))
        else:
            mag = (1.0 + m / float(2 ** mbits)) * float(2 ** (e - bias))
        rows.append((-mag if neg else mag, code, m))
    return rows


TABLES = {name: _table(name) for name in SPECS}


def _pick(name, scaled):
    """最近邻：绝对误差最小 → 符号一致（区分 ±0）→ 尾数位偶 → 码值小。"""
    neg = math.copysign(1.0, scaled) < 0.0
    best = None
    for val, code, m in TABLES[name]:
        sign_ok = 0 if ((code & 32) != 0) == neg else 1
        key = (abs(scaled - val), sign_ok, m & 1, code)
        if best is None or key < best[0]:
            best = (key, code, val)
    return best[1], best[2], best[0][0]


def _is_tie(name, scaled, best_dist):
    """两个**取值不同**的码点与 scaled 等距才算平局（同一数值的重复编码不算）。"""
    vals = {val for val, _c, _m in TABLES[name] if abs(scaled - val) == best_dist}
    return len(vals) >= 2


def _pack(codes):
    out = []
    for i in range(0, len(codes), 4):
        c0, c1, c2, c3 = codes[i:i + 4]
        out.append(c0 | ((c1 & 0x3) << 6))
        out.append((c1 >> 2) | ((c2 & 0xF) << 4))
        out.append((c2 >> 4) | (c3 << 2))
    return out


def _unpack(packed, n):
    codes = []
    for i in range(0, len(packed), 3):
        b0, b1, b2 = packed[i], packed[i + 1], packed[i + 2]
        codes.append(b0 & 0x3F)
        codes.append((b0 >> 6) | ((b1 & 0xF) << 2))
        codes.append((b1 >> 4) | ((b2 & 0x3) << 4))
        codes.append(b2 >> 2)
    return codes[:n]


def solve(values, fmt="auto"):
    if fmt not in ("auto", "e3m2", "e2m3"):
        raise ValueError("fmt 必须是 auto / e3m2 / e2m3")
    flat = values.to(torch.float64).reshape(-1)
    n = int(flat.numel())
    if bool(torch.isnan(flat).any()):
        raise ValueError("values 不得含 NaN（FP6 无 NaN 编码）")

    rows = flat.tolist()
    amax = 0.0
    for v in rows:
        a = abs(v)
        if a > amax:
            amax = a
    if fmt == "auto":
        if amax <= LIMITS["e2m3"]:
            fmt = "e2m3"
        elif amax <= LIMITS["e3m2"]:
            fmt = "e3m2"
        else:
            raise ValueError("amax 超出 FP6 可表示范围（最大 28）")

    ebits, mbits, _bias = SPECS[fmt]
    limit = LIMITS[fmt]
    max_code = ((2 ** ebits - 1) << mbits) | (2 ** mbits - 1)
    codes, dequant, sat_mask, tie_mask = [], [], [], []
    for v in rows:
        neg = math.copysign(1.0, v) < 0.0
        mag = abs(v)
        if mag > limit:
            code = max_code | (32 if neg else 0)
            codes.append(code)
            dequant.append(-limit if neg else limit)
            sat_mask.append(True)
            tie_mask.append(False)
            continue
        code, val, dist = _pick(fmt, -mag if neg else mag)
        codes.append(code)
        dequant.append(val)
        sat_mask.append(False)
        tie_mask.append(_is_tie(fmt, -mag if neg else mag, dist))

    pad = (-n) % 4
    packed = _pack(codes + [0] * pad) if n else []
    return {
        "fmt": fmt,
        "codes": torch.tensor(codes, dtype=torch.int64),
        "unpacked": torch.tensor(_unpack(packed, n), dtype=torch.int64),
        "packed": torch.tensor(packed, dtype=torch.int64),
        "dequant": torch.tensor(dequant, dtype=torch.float64),
        "sat_mask": torch.tensor(sat_mask, dtype=torch.bool),
        "tie_mask": torch.tensor(tie_mask, dtype=torch.bool),
        "pad_count": pad,
    }
