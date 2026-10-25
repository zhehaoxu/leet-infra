import torch


def solve(x_t, eps_pred, alpha_bar_t, alpha_bar_prev):
    """DDIM 确定性一步 (sigma=0)：x0_hat=(x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t);
    x_prev = sqrt(ab_prev)*x0_hat + sqrt(1-ab_prev)*eps。
    标量 alpha_bar_t,alpha_bar_prev in (0,1]；x_t,eps_pred 同形状。返回同形状 float32，无随机项。禁止现成扩散库。"""
    # TODO
    raise NotImplementedError
