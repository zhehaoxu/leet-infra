import torch


def solve(rewards, values, gamma, lam):
    """GAE：delta_t=r_t+gamma*V_{t+1}-V_t (V_T=0); 反向递推 A_t=delta_t+gamma*lam*A_{t+1};
    returns=A+values。rewards,values 形状 (T,)。返回 (advantages, returns) 两个 (T,) float32 张量。禁止现成 RL 库。"""
    # TODO
    raise NotImplementedError
