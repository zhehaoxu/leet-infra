import torch

# torch.manual_seed 无关：流程确定性。序列由参考解在评测环境实跑生成。
EXPECTED = [
    4.0, 2.56, 0.8464, 0.0154, 0.3809, 1.3476, 2.0106, 1.8653,
    1.0933, 0.3007, 0.0001, 0.2586, 0.7337, 0.997, 0.8584, 0.4576,
    0.0999, 0.0051, 0.1648, 0.3916,
]


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_sequence(solve):
    out = solve()
    assert isinstance(out, list) and all(isinstance(v, float) for v in out), \
        "必须返回 list[float]（loss.item() 记录，别返回张量）"
    assert len(out) == 20, f"expected 20 losses, got {len(out)}"
    for i, (g, e) in enumerate(zip(out, EXPECTED)):
        assert abs(g - e) < 1e-3, f"第 {i} 项 loss 不匹配: got {g!r}, expected {e!r}"


def check_len_and_first(solve):
    out = solve()
    assert len(out) == 20, f"length must be 20, got {len(out)}"
    assert abs(out[0] - 4.0) < 1e-6, f"第 0 项应为初始 w=2.0 的 loss 4.0, got {out[0]!r}"


def check_momentum_oscillation(solve):
    out = solve()
    # 动量过冲：第 3 项接近平底后回升，第 6 项附近达到第二个峰
    assert out[3] < 0.05, f"第 3 项应接近 0（首次过零点附近）, got {out[3]!r}"
    assert out[6] > 1.0, f"第 6 项应出现动量回升峰, got {out[6]!r}"


def check_decreasing_trend(solve):
    out = solve()
    assert out[-1] < out[0], "训练应收敛：末项 loss 应低于首项"
    assert max(out[10:]) < max(out[:8]), "后半程振荡幅度应比前半程衰减"


check_sequence.example = True
