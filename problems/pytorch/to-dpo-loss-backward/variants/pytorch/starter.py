import torch


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, mask_w, mask_l, beta):
    """DPO 损失的解析反向：给定逐 token 对数概率，返回损失与对两条 logp 的梯度。

    入参：
    - logp_w / logp_l: (B,T) float32/float64，偏好 / 被拒回答的**逐 token**
      对数概率（要梯度的那两个量）；
    - ref_logp_w / ref_logp_l: (B,) 与 logp 同 dtype，参考策略下的**序列级**
      对数概率（已按各自长度加总好，不参与求导）；
    - mask_w / mask_l: (B,T) bool，两条回答各自的有效 token；每条样本每侧
      至少一个 True；无效位置的梯度必须**严格为 0**（不许把 padding 的
      梯度也写成同一个数）；
    - beta: float > 0，DPO 温度。

    前向与反向（B 个样本取均值）：
    - 序列对数概率：s_w = Σ_t mask_w·logp_w，s_l = Σ_t mask_l·logp_l；
    - 隐式奖励差：Δ = (s_w - ref_w) - (s_l - ref_l)，logit x = β·Δ；
    - 损失：L = mean_b softplus(-x_b) = -mean_b log σ(x_b)；
    - 反向：dL/dx_b = -σ(-x_b)，dL/dΔ_b = -β·σ(-x_b)/B，
      于是
        grad_logp_w[b,t] = (dL/dΔ_b) · mask_w[b,t]
        grad_logp_l[b,t] = -(dL/dΔ_b) · mask_l[b,t]
      （两条回答对 Δ 的符号相反：提升被拒回答的概率会推高损失）。

    返回 dict（键固定）：
    - 'loss': 0 维 float64；
    - 'grad_logp_w'、'grad_logp_l': (B,T) float64，已含 1/B 的平均缩放。

    非法输入抛 ValueError：非张量 / logp 非二维或形状不符 / 非浮点 /
    含 NaN·Inf / ref 形状不是 (B,) 或 dtype 不符 / mask 非 bool 或形状不符 /
    某条样本某一侧没有有效 token / beta <= 0。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
