import torch


def solve(logp, logp_old, ref_logp, mask, adv, eps_low=0.2, eps_high=0.2):
    """CISPO 策略损失（停梯度裁剪权重）+ K3 形式的 KL 估计。

    入参：
    - logp / logp_old / ref_logp: (N,T) float32/float64，当前策略、行为策略、
      参考策略的逐 token 对数概率；
    - mask: (N,T) bool，有效 token，至少一个 True；
    - adv: (N,) 与 logp 同 dtype，每条回答一个序列级优势；
    - eps_low: float ∈ [0,1)，下界半径；eps_high: float >= 0，上界半径。

    策略损失（CISPO，MiniMax-M1）：
    - 重要性比 rho = exp(logp - logp_old)；
    - 裁剪权重 w = clamp(rho, 1-eps_low, 1+eps_high)，**对 w 停梯度**
      （把它当常数，梯度只穿过 logp 本身）：
        L = -Σ mask·w·A·logp / N_valid，N_valid = Σ mask。
      这与 PPO 的 min(rho·A, clip·A) 不同：PPO 的梯度会穿过比值，
      CISPO 只保留「裁剪后的标量权重 × logπ」这一条路径。

    KL 估计（K3，Schulman 的 k3 估计器）：
    - 令 log_r = ref_logp - logp（注意顺序：**参考减当前**，比值为
      π_ref/π，样本来自当前策略 π）；
    - k3 = exp(log_r) - log_r - 1 >= 0，逐 token 无偏估计 KL(π‖π_ref)；
    - kl_k3 = Σ mask·k3 / N_valid。
    - 为保证 exp 不溢出，要求 max(ref_logp - logp) <= 20，否则 ValueError。

    返回 dict（键固定）：
    - 'loss': 0 维 float64；
    - 'kl_k3': 0 维 float64；
    - 'clip_frac': 0 维 float64，rho 落在 [1-eps_low, 1+eps_high] **之外**
      的有效 token 占比（严格不等号判定）。

    非法输入抛 ValueError：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 /
    N 或 T < 1 / mask 非 bool / 无有效 token / adv 形状或 dtype 不符 /
    eps_low 不在 [0,1) / eps_high < 0 / log_r 超过 20。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
