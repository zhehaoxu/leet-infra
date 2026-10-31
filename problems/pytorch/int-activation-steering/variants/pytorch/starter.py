import torch


def solve(hiddens, s, alphas, layer_indices, normalize=False):
    """转写 steering vector 注入（nrimsky/steering-vectors 口径）的确定性账本。

    参数
    ----
    hiddens : [L, T, D]  逐层逐 token 的残差流隐态
    s       : [D]        steering vector
    alphas  : list[float]（长度 K ≥ 1）每次注入的系数
    layer_indices : list[int]（长度 K）每次注入的层号，可重复
    normalize : bool  True 时先把 s 归一化为 s/|s| 再注入（所有次共用同一 s_eff）

    注入（按 k = 0..K-1 顺序执行）：
        steered[layer_indices[k]] += alphas[k] * s_eff
    同层多次注入是**累加**（不是覆盖）。

    返回
    ----
    dict：
        steered        [L, T, D]  注入后的隐态（原张量不被修改）
        s_norm         float      |s|（归一化前）
        layer_delta    list[float]（长 L）每层累计注入向量的范数 |Σ_k α_k|·|s_eff|，
                       未注入的层为 0.0
        cos_before     [L][T]     注入前每层每 token 的 cos(h, s)
        cos_after      [L][T]     注入后每层每 token 的 cos(h', s)
        proj_coef      [L][T]     每层每 token 的 (h·s)/(s·s)——使 h' ⊥ s 的 α

    cos/分子分母一律 float64 语义；同方向注入保证 layer_delta 可加。
    """
    raise NotImplementedError
