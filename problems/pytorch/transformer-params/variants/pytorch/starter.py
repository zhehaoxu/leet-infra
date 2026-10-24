# 按 Transformer 结构清点参数量
def solve(layers, hidden, ffn, vocab, tied_embedding=True):
    # TODO: P_layer = 4*h^2 + 2*h*f + 4*h；加上嵌入(×1或×2) 与 final LN(2h)
    raise NotImplementedError
