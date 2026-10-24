# Transformer 参数量估算参考解
def solve(layers, hidden, ffn, vocab, tied_embedding=True):
    per_layer = 4 * hidden * hidden + 2 * hidden * ffn + 4 * hidden
    total = layers * per_layer + vocab * hidden
    if not tied_embedding:
        total += vocab * hidden
    total += 2 * hidden  # final LayerNorm
    return int(total)
