import torch


def _ref(b, s, layers, hidden, ffn, heads, dtype_bytes=2):
    """独立重算参考（不复用 solve 的实现路径），核对字节数。"""
    e_full = b * s * (6 * hidden + 2 * ffn) + b * heads * s * s
    full = layers * e_full * dtype_bytes
    ckpt = (layers * b * s * hidden + e_full) * dtype_bytes
    return full, ckpt


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    full, ckpt = solve(4, 1024, 6, 512, 2048, 8, 2)
    assert (full, ckpt) == (754974720, 150994944), (full, ckpt)


def check_matches_ref(solve):
    for args in [(2, 256, 4, 128, 512, 4, 2),
                 (8, 4096, 12, 1024, 4096, 16, 2),
                 (1, 512, 3, 256, 1024, 8, 4)]:
        b, s, layers, h, f, a, dt = args
        assert tuple(solve(*args)) == _ref(*args), args


def check_checkpoint_reduces(solve):
    # 检查点应显著低于完整保存，且随层数增大差距更明显
    full, ckpt = solve(4, 1024, 12, 768, 3072, 12, 2)
    assert ckpt < full
    assert full // ckpt >= 3, (full, ckpt)


def check_returns_ints(solve):
    full, ckpt = solve(2, 128, 2, 64, 128, 4)
    assert isinstance(full, int) and isinstance(ckpt, int), (type(full), type(ckpt))


check_example.example = True
