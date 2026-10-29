def solve(hidden, embeddings, W, eps=1e-6):
    """MTP 的多深度预测：逐层把「上一层隐状态 + 下一个 token 的输入嵌入」融合再投影。

    hidden:(B, S, d)     主干最后一层的隐状态 h^0
    embeddings:(B, L, d) 输入嵌入序列，L 必须至少 S + D（要能看到向前 D 个 token）
    W:(D, 2d, d)         第 k 层（k 从 1 计）的融合投影，作用在 [rms(h^{k-1}); rms(Emb_{:,k:k+S})]
    eps: RMSNorm 的数值稳定项

    递推：
      h^0 = hidden
      h^k = W_k · [ RMS(h^{k-1}) ; RMS(embeddings[:, k : k+S]) ]     k = 1..D

    RMS 沿最后一维（特征维）计算。返回长度为 D 的 tuple，元素为 float32 的 (B, S, d)，
    依次是 h^1 .. h^D（不包含输入的 h^0）。
    """
    # TODO
    raise NotImplementedError
