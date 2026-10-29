import torch


def solve(scores, selected_ids, alpha=1.0):
    """中文契约 / English contract
    scores [B,T,E]：非负亲和度而非 logits，每个 token 的专家和 >= 1e-6 /
        nonnegative affinities, not logits; each token's expert sum is >= 1e-6.
    selected_ids [B,T,K] int64：每 token 的 K 个专家互不重复；选择由外部给定，不得重算 /
        K distinct expert IDs per token; selections are supplied and must not be recomputed.
    B=1..4, T,E=1..8, 1<=K<=E；alpha 在 [0,2]；CPU float64，不修改输入，接受非连续张量 /
        B=1..4, T,E=1..8, 1<=K<=E; alpha in [0,2]; CPU float64, mutate nothing, accept noncontiguous tensors.
    先在每个 token 内对全 E 个专家归一化；f=E*count/(T*K)；每条序列先算 alpha*sum(f*p) 再跨 batch 平均 /
        normalize over all E experts per token; f=E*count/(T*K); compute alpha*sum(f*p) per sequence,
        then average across the batch.
    返回 / Returns (f [B,E], p [B,E], per_sequence [B], sequence_loss scalar, batch_loss scalar)；
        batch_loss 先跨 batch 平均 f、p 再点积 / batch_loss averages f and p across the batch first.
    """
    raise NotImplementedError
