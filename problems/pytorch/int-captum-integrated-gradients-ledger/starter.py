CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 本题是 Captum Integrated Gradients（pytorch/captum）的四段纯数值账：
# 求积参数（step_sizes / alphas）、scaled_features 的官方 step-major 布局、
# 归因聚合（step_sizes 加权 + 沿 step 维求和 + multiply_by_inputs），
# 以及完备性公理 delta（attr_sum - (end_out_sum - start_out_sum)）。
# 不做前向传播、不做自动微分、不做 GPU kernel。
#
# mode 列表：
#   quadrature_params  —— 入参 method（riemann_left / riemann_right /
#                          riemann_middle / riemann_trapezoid / gausslegendre）
#                          / n_steps
#                          返回 step_sizes / alphas
#   scaled_features    —— 入参 input（[B][F]）/ baseline（[B][F]）/ method /
#                          n_steps
#                          返回 scaled_features（[S*B][F]，官方 step-major 布局）
#   ig_aggregate       —— 入参 input（[B][F]）/ baseline（[B][F]）/ grads
#                          （[S*B][F]，官方 flatten 后的 step-major 布局）/
#                          method / n_steps / multiply_by_inputs（默认 True）
#                          返回 total_grads / attributions（均为 [B][F]）
#   ig_completeness    —— 入参 attributions（[B][F]）/ start_out（[B][K]）/
#                          end_out（[B][K]）
#                          返回 attr_sum / start_out_sum / end_out_sum / delta
#
# 口径：Riemann 四变体的 step_sizes 与 alphas 一律按**精确有理数**计算，
# 输出转 float 后逐位相等判定；Gauss-Legendre 含无理数根，用相对容差判定。
#
# 所有 mode 都返回 dict，字段含义见 description.md。
def solve(mode: str, **kw):
    raise NotImplementedError
