def _ref(layers, heads, head_dim, seq, batch, dtype_bytes=2, kv_heads=None):
    """独立重算：先算单头单 token 单层的 K+V 字节，再逐层逐头累加。"""
    kvh = heads if kv_heads is None else kv_heads
    per_kv_elem = head_dim * dtype_bytes          # 一个头一个 token 一层的一份额
    per_token = 2 * layers * kvh * per_kv_elem     # K 与 V，跨所有层
    return per_token * seq * batch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    assert solve(32, 32, 128, 2048, 1, 2) == 1073741824
    assert solve(32, 32, 128, 2048, 1, 2, kv_heads=1) == 33554432


def check_matches_ref(solve):
    cases = [
        (12, 12, 64, 512, 4, 2, None),
        (24, 32, 128, 4096, 2, 2, 8),
        (32, 40, 128, 8192, 1, 1, 1),
        (8, 16, 96, 1024, 8, 4, None),
        (2, 4, 32, 0, 1, 2, None),
    ]
    for c in cases:
        assert solve(*c) == _ref(*c), c


def check_mqa_gqa_ratios(solve):
    base = solve(32, 32, 128, 2048, 1, 2)                 # MHA
    mqa = solve(32, 32, 128, 2048, 1, 2, kv_heads=1)
    gqa = solve(32, 32, 128, 2048, 1, 2, kv_heads=8)
    assert base == mqa * 32, (base, mqa)                  # MQA 缩到 1/heads
    assert base == gqa * 4, (base, gqa)                   # GQA(8) 缩到 1/4


def check_linear_in_seq_batch(solve):
    a = solve(8, 8, 64, 100, 1, 2)
    b = solve(8, 8, 64, 300, 1, 2)
    c = solve(8, 8, 64, 100, 3, 2)
    assert b == a * 3 and c == a * 3, (a, b, c)
    assert solve(8, 8, 64, 0, 5, 2) == 0


def check_returns_int(solve):
    assert isinstance(solve(4, 4, 64, 128, 2), int)


check_example.example = True
