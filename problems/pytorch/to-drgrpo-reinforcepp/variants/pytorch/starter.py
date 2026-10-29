import torch


def solve(rewards, logp, logp_old, mask, eps=0.2):
    """同一批 rollout 上算两种优势与两种损失：Dr. GRPO 与 REINFORCE++。

    入参：
    - rewards: (B,G) float，B 个 prompt、每个 G >= 2 条回答的标量奖励；
    - logp / logp_old: (B,G,T) float，当前策略与行为策略的逐 token 对数概率；
    - mask: (B,G,T) bool，有效 token 位置；每条回答至少一个 True；
    - eps: float ∈ (0,1)，PPO 对称裁剪半径（两边都用 1±eps，只比归一化）。

    优势：
    - Dr. GRPO：只在**组内中心化**，不除标准差：
        A_dr[b,i] = r[b,i] - mean_j r[b,j]。
    - REINFORCE++：在**整个 batch** 上白化（先减全局均值，再除**总体**
      标准差，即 std(unbiased=False)）：
        A_rp[b,i] = (r[b,i] - mean_all) / std_all；
      若 std_all <= 1e-12（视为 0），则整批 adv 取 0（不做除零）。

    逐 token 重要性比 rho = exp(logp - logp_old)，对称裁剪
    clip = clamp(rho, 1-eps, 1+eps)，surrogate = min(rho·A, clip·A)。

    损失（差别只在缩放次序）：
    - Dr. GRPO：把整批**所有 token** 加总，除以常数 B·G·T（T 是张量宽度
      L_max，**不是**每条回答的有效长度，也不做逐条平均）：
        L_dr = -Σ_{b,i,t} mask·surrogate / (B·G·T)。
    - REINFORCE++：每条回答先按**自身有效 token 数**取均值，再对 B·G 条
      回答取均值：
        L_rp = -mean_{b,i} ( Σ_t mask·surrogate / |o_{b,i}| )。

    返回 dict（键固定）：
    - 'adv_drgrpo' (B,G) float64；'adv_reinforcepp' (B,G) float64；
    - 'loss_drgrpo'、'loss_reinforcepp' 均为 0 维 float64 标量。

    非法输入抛 ValueError：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 /
    G < 2 / T < 1 / mask 非 bool / 有空回答 / eps 不在 (0,1)。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
