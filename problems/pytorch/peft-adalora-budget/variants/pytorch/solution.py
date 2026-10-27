import torch


def solve(E, A, B, g1_E, g1_A, g1_B, g2_E, g2_A, g2_B, beta1, beta2, budget):
    """AdaLoRA 一次预算重分配：两步 EMA 敏感度评分 + 全局第 k 小截断。
    E:(r,1) A:(r,d_in) B:(d_out,r)，梯度同形；返回 (scores:(r,), masked_E:(r,1))，float32。
    掩码只置零 E 中 score <= 第 k 小（k=r-budget，k<1 不掩码）的位置。禁止现成 PEFT 库。"""
    b1, b2 = float(beta1), float(beta2)

    def element_scores(p, g1, g2):
        ipt1 = (p.to(torch.float64) * g1.to(torch.float64)).abs()
        ipt2 = (p.to(torch.float64) * g2.to(torch.float64)).abs()
        avg = (1.0 - b1) * ipt1
        unc = (1.0 - b2) * (ipt1 - avg).abs()
        avg = b1 * avg + (1.0 - b1) * ipt2
        unc = b2 * unc + (1.0 - b2) * (ipt2 - avg).abs()
        return avg * unc

    sE = element_scores(E, g1_E, g2_E).view(-1)
    sA = element_scores(A, g1_A, g2_A).mean(dim=1)
    sB = element_scores(B, g1_B, g2_B).mean(dim=0)
    scores = sE + sA + sB
    masked_E = E.to(torch.float64).clone()
    k = scores.numel() - int(budget)
    if k >= 1:
        thresh = torch.kthvalue(scores, k).values
        masked_E[scores <= thresh, 0] = 0.0
    return (scores.to(torch.float32), masked_E.to(torch.float32))
