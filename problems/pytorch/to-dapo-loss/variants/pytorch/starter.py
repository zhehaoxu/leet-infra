import torch


def solve(logp, logp_old, mask, rewards, eps_low=0.2, eps_high=0.28):
    """DAPO 损失：非对称 clip-higher + 跨 batch 的 token 级聚合 + 动态组过滤。

    入参：
    - logp / logp_old: (B,G,T) float，当前策略与行为策略的逐 token 对数概率；
    - mask: (B,G,T) bool，有效 token；每条回答至少一个 True；
    - rewards: (B,G) float，元素必须**严格是 0 或 1**（正确性标记）；
    - eps_low: float ∈ [0,1)，下界裁剪半径（1-eps_low）；
    - eps_high: float >= 0，上界裁剪半径（1+eps_high），允许 != eps_low。

    步骤（顺序敏感）：
    1. 组内优势按 GRPO 口径归一：
         A[b,i] = (r[b,i] - mean_j r[b,j]) / std_j r[b,j]，
       std 用**总体**标准差（unbiased=False）；若该组 std == 0，则整组 A 取 0。
    2. 动态采样组过滤：组内正确率 acc[b] = mean_j r[b,j]，
       **只保留 0 < acc < 1 的组**（全对 / 全错的组对策略梯度无信号，
       直接丢弃）；keep 是 (B,) bool。
    3. 逐 token 重要性比 rho = exp(logp - logp_old)，
       **非对称**裁剪 clipped = clamp(rho, 1-eps_low, 1+eps_high)；
       surrogate = min(rho·A, clipped·A)。
    4. token 级聚合：只在保留组内的有效 token 上求**一个全局平均**：
         L = -Σ_{b,i,t} valid·surrogate / N_valid，valid = mask & keep；
       若没有任何保留 token，L = 0（不返回 NaN）。
       注意这里不是「每条回答先平均再对回答平均」。

    返回 dict（键固定）：
    - 'loss': 0 维 float64 标量；
    - 'adv': (B,G) float64；
    - 'keep': (B,) bool；
    - 'num_tokens': 0 维 int64，保留组内有效 token 总数。

    非法输入抛 ValueError：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 /
    G < 2 / T < 1 / rewards 非 0/1 / mask 非 bool / 有空回答 /
    eps_low 不在 [0,1) / eps_high < 0。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
