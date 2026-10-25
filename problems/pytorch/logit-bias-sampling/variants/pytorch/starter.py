import torch


def solve(logits, bias_tokens, bias_values, banned, u):
    # TODO: logit-bias + banned tokens（u 为显式输入）。
    # ① 按列表顺序逐条 logit[token] += bias（同一 token 多次出现则累加；越界/负 id 的条目无副作用）；
    # ② banned 的 token 置 -inf（强制掩码）；
    # ③ 对偏置后的 logits 做 softmax，被 ban 项概率恰为 0，其余按存活集合归一化；
    # ④ u 在 filtered 原顺序 CDF 上逆变换采样得 chosen。
    # 返回 (biased_logits, filtered, chosen)，biased_logits 形状同 logits、含 -inf。
    raise NotImplementedError
