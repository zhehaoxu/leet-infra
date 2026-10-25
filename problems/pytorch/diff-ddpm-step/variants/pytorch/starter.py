import torch


def solve(x_t, x0_pred, beta_t, alpha_bar_t, alpha_bar_prev, z):
    """DDPM 反向后验一步：a_t=1-beta_t;
    c_t=sqrt(a_t)*(1-ab_prev)/(1-ab_t); c_0=sqrt(ab_prev)*beta_t/(1-ab_t);
    var=beta_t*(1-ab_prev)/(1-ab_t); x_prev=c_t*x_t + c_0*x0_pred + sqrt(var)*z。
    标量 beta_t,alpha_bar_t,alpha_bar_prev；x_t,x0_pred,z 同形状。返回同形状 float32。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
