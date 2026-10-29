import torch


def solve(student_logits, teacher_logits, labels, lam=0.5, beta=0.5,
          temperature=1.0, ignore_index=-100):
    """GKD 损失：(1-λ)·广义 JSD + λ·硬标签交叉熵，均在有效 token 上取均值。

    入参：
    - student_logits: (N, V) float32/float64，学生未归一 logits；
    - teacher_logits: (N, V) 同形状同 dtype，教师未归一 logits；
    - labels: (N,) int32/int64，每个位置的硬标签；等于 ignore_index 的位置
      完全不参与两个损失项（既不进 JSD 均值也不进 CE 均值）；
    - lam: float ∈ [0, 1]，CE 项权重 λ（GKD 默认 0.5）；
    - beta: float，必须**严格**落在 (0, 1)，广义 JSD 的插值系数；
    - temperature: float > 0，蒸馏温度 τ，只作用于 JSD 的两条分布；
    - ignore_index: int，默认 -100。

    口径（顺序敏感，按此实现）：
    1. 温度缩放后再取对数概率：
         sl = log_softmax(student_logits / τ)，tl = log_softmax(teacher / τ)；
       p_s = exp(sl)，p_t = exp(tl)。
    2. 混合分布按**概率**定义：m = β·p_t + (1-β)·p_s，等价的对数式
         log m = logsumexp( stack([ tl + log β,  sl + log(1-β) ]) )；
    3. 广义 JSD（GKD 论文的 skew-JS 口径）：
         JSD = β·KL(p_t ‖ m) + (1-β)·KL(p_s ‖ m)
             = β·Σ_v p_t (tl - log m) + (1-β)·Σ_v p_s (sl - log m)。
       β→0 时 JSD/β → KL(p_t‖p_s)（前向 KL），β→1 时 JSD/(1-β) → KL(p_s‖p_t)
       （反向 KL）；β 恰为 0 或 1 时上式整体退化为 0，故契约要求开区间。
    4. 交叉熵用**未缩放**的学生 logits：
         CE = -log_softmax(student_logits)[label]。
    5. 总损失 = (1-λ)·mean_valid(JSD) + λ·mean_valid(CE)，
       不乘 τ²（GKD 论文与 TRL 实现均无此因子）；返回 0 维 float64 标量。

    非法输入抛 ValueError：非张量 / logits 非二维或形状不符 / 非浮点 /
    含 NaN·Inf / V < 1 / labels 非整型或形状不符 / 有效标签越界 /
    无有效标签 / lam 越界 / beta 不在开区间 (0,1) / temperature <= 0。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
