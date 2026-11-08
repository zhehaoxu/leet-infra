CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 本题是 librosa YIN 的三段纯数值账：CMND 累积均值归一化差分、抛物线插值偏移、
# 谷值挑选与绝对阈值。三段都不做 FFT 自相关，也不读真音频波形。
#
# mode 列表：
#   cmnd              —— 入参 sq / acf / min_period / max_period / tiny
#                        返回 difference / cumulative_mean / denominator / cmnd
#   parabolic_shifts  —— 入参 x
#                        返回 a / b / shifts
#   trough_selection  —— 入参 x / min_period / sr / trough_threshold
#                        返回 is_trough / is_threshold_trough / global_min /
#                        yin_period / no_trough_below_threshold / shift / period / f0
#
# 所有 mode 都返回 dict，字段含义见 description.md。
def solve(mode: str, **kw):
    raise NotImplementedError
