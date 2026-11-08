"""order 838 参考解（暂存）：librosa YIN 的 CMND / 抛物线插值 / 绝对阈值账。

官方回源：librosa/librosa (main)
  librosa/core/pitch.py::_cumulative_mean_normalized_difference
  librosa/core/pitch.py::_parabolic_interpolation / _pi_stencil
  librosa/util/utils.py::localmin / _localmin_sten

三段纯数值账，不做 FFT 自相关、不做真音频 IO、不依赖 numpy/librosa/numba。
"""

import math

# numpy.float64 的最小正规数；官方 util.tiny() 返回的就是它。
TINY = 2.2250738585072014e-308


def _cmnd(sq, acf, min_period, max_period, tiny):
    """Cumulative mean normalized difference（官方 _cumulative_mean_normalized_difference）。

    sq   : y^2 序列，长度 >= max_period
    acf  : 自相关 r[0..max_period]
    返回 d / cumulative_mean / denominator / cmnd 四条账
    """
    # Energy terms: E[k] = sum_{m<=k} y_m^2
    energy = []
    acc = 0.0
    for v in sq:
        acc += v
        energy.append(acc)

    # 官方 quirk：写 d 之前先把 E[0] 抹成 0。
    energy[0] = 0.0

    # Difference function: d(k) = 2 * (ACF(0) - ACF(k)) - E[k-1]
    difference = [0.0] * (max_period + 1)
    for k in range(1, max_period + 1):
        difference[k] = 2.0 * (acf[0] - acf[k]) - energy[k - 1]

    # Cumulative mean: 分母是 k_range = [1, 2, ..., max_period]，不是 tau。
    cumulative_mean = []
    running = 0.0
    for j in range(max_period):
        running += difference[j + 1]
        cumulative_mean.append(running / (j + 1))

    # 分母切片下标偏移 min_period - 1，长度 max_period - min_period + 1。
    denominator = cumulative_mean[min_period - 1:max_period]

    numerator = difference[min_period:max_period + 1]
    cmnd = []
    for i in range(len(numerator)):
        cmnd.append(numerator[i] / (denominator[i] + tiny))

    return {
        "difference": difference,
        "cumulative_mean": cumulative_mean,
        "denominator": denominator,
        "cmnd": cmnd,
    }


def _parabolic(x):
    """Piecewise parabolic interpolation（官方 _parabolic_interpolation / _pi_stencil）。

    a = x[i+1] + x[i-1] - 2 x[i]，b = (x[i+1] - x[i-1]) / 2；
    |b| >= |a| -> 0，否则 -b / a；首尾 bin 强制 0。
    """
    n = len(x)
    a_list = [0.0] * n
    b_list = [0.0] * n
    shifts = [0.0] * n
    for i in range(1, n - 1):
        a_val = x[i + 1] + x[i - 1] - 2.0 * x[i]
        b_val = (x[i + 1] - x[i - 1]) / 2.0
        a_list[i] = a_val
        b_list[i] = b_val
        if abs(b_val) >= abs(a_val):
            shifts[i] = 0.0
        else:
            shifts[i] = -b_val / a_val
    # stencil 覆盖不到的边界：官方显式置 0
    shifts[0] = 0.0
    shifts[n - 1] = 0.0
    return {"a": a_list, "b": b_list, "shifts": shifts}


def _trough_selection(x, min_period, sr, trough_threshold):
    """官方 yin() 的 localmin -> 绝对阈值 -> 全局最小回退 -> period/f0 段。"""
    n = len(x)
    shifts = _parabolic(x)["shifts"]

    # util.localmin：内部 (x[i] < x[i-1]) and (x[i] <= x[i+1])；
    # 末元素 x[-1] < x[-2]；首元素 stencil 边界恒 False。
    is_trough = [False] * n
    for i in range(1, n - 1):
        is_trough[i] = (x[i] < x[i - 1]) and (x[i] <= x[i + 1])
    is_trough[n - 1] = x[n - 1] < x[n - 2]

    # 官方覆盖第 0 个 tau bin
    is_trough[0] = x[0] < x[1]

    # 绝对阈值：严格小于
    is_threshold_trough = [
        bool(is_trough[i]) and bool(x[i] < trough_threshold) for i in range(n)
    ]

    # argmin / argmax 并列取第一个
    global_min = 0
    for i in range(1, n):
        if x[i] < x[global_min]:
            global_min = i

    yin_period = -1
    for i in range(n):
        if is_threshold_trough[i]:
            yin_period = i
            break
    if yin_period < 0:
        no_trough = True
        yin_period = global_min
    else:
        no_trough = False

    shift = shifts[yin_period]
    period = min_period + yin_period + shift
    f0 = sr / period

    return {
        "is_trough": is_trough,
        "is_threshold_trough": is_threshold_trough,
        "global_min": global_min,
        "yin_period": yin_period,
        "no_trough_below_threshold": no_trough,
        "shift": shift,
        "period": period,
        "f0": f0,
    }


def solve(mode: str, **kw):
    if mode == "cmnd":
        return _cmnd(
            list(kw["sq"]),
            list(kw["acf"]),
            int(kw["min_period"]),
            int(kw["max_period"]),
            float(kw.get("tiny", TINY)),
        )
    if mode == "parabolic_shifts":
        return _parabolic(list(kw["x"]))
    if mode == "trough_selection":
        return _trough_selection(
            list(kw["x"]),
            int(kw["min_period"]),
            float(kw["sr"]),
            float(kw["trough_threshold"]),
        )
    raise ValueError("unknown mode: %s" % mode)
