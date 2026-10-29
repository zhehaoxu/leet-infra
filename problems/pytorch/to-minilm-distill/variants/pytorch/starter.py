import torch


def solve(teacher_attn, student_attn, teacher_cls, student_cls, mask,
          temperature=1.0):
    """MiniLM 蒸馏损失：逐头注意力分布 KL + CLS 三元组 MSE，两项直接相加。

    入参：
    - teacher_attn / student_attn: (B,H,L,L) float32/float64，**未归一**的
      注意力打分（softmax 之前）；最后一维是 key，倒数第二维是 query。
    - mask: (B,L) bool，同一个位置既是合法 query 也是合法 key；
      False 的位置既不能当 query 计损失，也不能当 key 被注意到。
    - teacher_cls / student_cls: (B,3,D) float，三对 CLS 表示
      （如 CLS·CLS、CLS·token、token·token 三种关系表示），逐元素 MSE。
    - temperature: float > 0，注意力 softmax 的温度 τ。

    口径（顺序敏感）：
    1. 注意力分布：logits/τ 后，在 key 维（最后一维）做 masked softmax；
       mask=False 的 key 概率**严格为 0**（用 -inf 填充，再取 log_softmax）。
    2. 逐 (b,h,i) 的 KL：KL(p_t ‖ p_s) = Σ_j p_t (log p_t - log p_s)，
       其中 p_t、p_s 各自按自己的 softmax 归一（两边同一个 mask）。
    3. attn_kl = Σ_{有效 query} Σ_h KL / (有效 query 数 × H)。
       无效 query（mask=False）的行整行不计入，也不做任何平均。
    4. cls_mse = mean( (teacher_cls - student_cls)² )，对全部 B×3×D 元素取均值。
    5. 返回 dict：{'attn_kl', 'cls_mse', 'loss'}，三个都是 0 维 float64，
       且 loss = attn_kl + cls_mse（权重 1:1）。

    非法输入抛 ValueError：非张量 / 注意力非四维或非方阵 / 非浮点 / 含
    NaN·Inf / 形状不一致 / mask 非 bool 或形状不符 / mask 全 False /
    CLS 形状不是 (B,3,D) 或与 B 不符 / D < 1 / temperature <= 0。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
