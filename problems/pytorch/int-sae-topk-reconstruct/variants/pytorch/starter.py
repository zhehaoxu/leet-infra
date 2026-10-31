import torch
import torch.nn.functional as F


def solve(x, W_enc, b_enc, W_dec, b_dec, k):
    """转写 TopK SAE 的 encode/decode 账（SAELens 的 TopK 架构，隐态全显式入参）。

    参数
    ----
    x     : [B, D_in]   输入隐态
    W_enc : [F, D_in]   编码矩阵（每行一个特征方向）
    b_enc : [F]
    W_dec : [F, D_in]   解码矩阵（与 W_dec 行 = 特征方向；与 W_enc 不绑定时彼此独立）
    b_dec : [D_in]
    k     : int，保留的最大特征数（1 <= k <= F）

    返回
    ----
    dict：
        feature_acts   [B, F]   top-k 稀疏激活（非 top-k 位置严格为 0）
        recon          [B, D_in] (feature_acts @ W_dec) + b_dec
        recon_error    [B, D_in] x - recon
        l0             [B]       每个样本的非零激活计数 = min(k, pre_acts>0 计数)
        mse            [B]       recon_error 逐行均方（对 D_in 求均值）
        max_feat       [B]       每行激活绝对值最大的特征 index（并列取较小 index）

    步骤（float 化计算）：
        pre_acts = (x - b_dec) @ W_enc.T + b_enc
        取每行值最大的 k 个（并列时取较小 index，torch.topk 语义），
        激活值保留原值（可为负——上游 TopK 不做 ReLU，负的 pre-act 也会入选），
        其余位置清零
        recon = acts @ W_dec + b_dec
    """
    raise NotImplementedError
