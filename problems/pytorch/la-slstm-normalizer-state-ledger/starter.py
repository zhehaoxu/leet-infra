CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 本题是 xLSTM sLSTM 的三段纯数值账：max 稳定器与门控、标量记忆 + normalizer
# state 的点式递推、以及 f 门 bias 的两种初始化。三段都不做 GEMM、不做反向传播。
#
# mode 列表：
#   max_stabilizer  —— 入参 iraw / fraw / m_prev / n_prev
#                      返回 logfplusm / m_new / zero_n_branch / igate / fgate / gate_le_one
#   pointwise_step  —— 入参 wx / ry / bias / states / head_dim
#                      返回 raw_gates / logfplusm / m_new / zero_n_branch / igate /
#                      fgate / ogate / zval / c_new / n_new / y_new
#   bias_powerlaw   —— 入参 head_dim / block_idx / num_blocks（可选 bias_mode）
#                      返回 bias / exponent / ratio
#
# 所有 mode 都返回 dict，字段含义见 description.md。
def solve(mode: str, **kw):
    raise NotImplementedError
