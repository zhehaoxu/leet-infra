def _ref(layers, hidden, ffn, vocab, tied):
    """独立清点参考（逐矩阵累加，不复用 solve 的公式路径），交叉核对参数量。"""
    attn = 4 * (hidden * hidden)               # q, k, v, o
    ffn_p = hidden * ffn + ffn * hidden        # 两个线性层
    ln = 2 * (hidden + hidden)                 # 每层 2 个 LayerNorm 的 weight+bias
    total = layers * (attn + ffn_p + ln)
    total += vocab * hidden                    # 输入嵌入
    if not tied:
        total += vocab * hidden                # 独立的输出投影
    total += hidden + hidden                   # final LayerNorm
    return total


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    assert solve(12, 768, 3072, 50257, True) == 123570432


def check_matches_ref(solve):
    for args in [(2, 128, 512, 300, True), (6, 256, 1024, 32000, False),
                 (8, 320, 640, 4096, True), (3, 96, 384, 64, False),
                 (12, 768, 3072, 50257, False)]:
        assert solve(*args) == _ref(*args), args


def check_untied_adds_head(solve):
    tied = solve(4, 512, 2048, 1000, True)
    untied = solve(4, 512, 2048, 1000, False)
    assert untied - tied == 1000 * 512, (tied, untied)


def check_minimal_layer(solve):
    # layers=1, h=2, f=3, vocab=4, tied -> 36 + 8 + 4 = 48
    assert solve(1, 2, 3, 4, True) == 48


def check_returns_int(solve):
    v = solve(2, 64, 128, 500, True)
    assert isinstance(v, int), type(v)


check_example.example = True
