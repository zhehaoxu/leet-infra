"""xLSTM mLSTM stabilized 指数门与 D 矩阵账（starter）。

实现 solve(mode, **kw)，通过全部用例。模式清单：
  stabilized_step    m_state 的 log-sum-exp 稳定化 + fg/ig 双门激活 + C/n 更新 + h_denom 下界
  parallel_decay     cumsum 差分构造 log_fg_matrix / log_D_matrix / rowwise 稳定化 / normalizer
  chunk_intra_gates  chunk 内 loggates / m_loc / kv / ksum（不判 chunk 间递推与三形式恒等）

各 mode 的参数语义见题面「模式与规则」；logsigmoid = -softplus(-x)，eps 一律加在分母外。
判题环境只有 torch + numpy；本题为纯 Python 数值账，不跑真实模型、不加载权重、不 import xlstm。
"""

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def solve(mode: str, **kw):
    raise NotImplementedError
