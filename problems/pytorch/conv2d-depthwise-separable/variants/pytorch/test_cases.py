import torch
import torch.nn.functional as F


def _ref(x, dw, pw, dwb, pwb):
    y = F.conv2d(x, dw, dwb, padding=dw.shape[2] // 2, groups=x.shape[1])
    return F.conv2d(y, pw, pwb)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.ones(1, 2, 3, 3)
    dw = torch.ones(2, 1, 3, 3)
    pw = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1)
    y = solve(x, dw, pw)
    exp = torch.tensor(
        [[[[4, 6, 4], [6, 9, 6], [4, 6, 4]],
          [[20, 30, 20], [30, 45, 30], [20, 30, 20]],
          [[36, 54, 36], [54, 81, 54], [36, 54, 36]]]], dtype=torch.float32)
    assert torch.allclose(y, exp), f"expected {exp.tolist()}, got {y.tolist()}"


def check_matches_two_step_conv(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, 6)
    dw = torch.randn(3, 1, 3, 3)
    pw = torch.randn(5, 3, 1, 1)
    dwb = torch.randn(3)
    pwb = torch.randn(5)
    assert torch.allclose(solve(x, dw, pw, dwb, pwb), _ref(x, dw, pw, dwb, pwb), atol=1e-5)


def check_5x5_depthwise(solve):
    torch.manual_seed(1)
    x = torch.randn(1, 4, 7, 7)
    dw = torch.randn(4, 1, 5, 5)      # kh=5, padding=2
    pw = torch.randn(2, 4, 1, 1)
    y = solve(x, dw, pw)
    assert y.shape == (1, 2, 7, 7), "padding=kh//2 应保持空间尺寸"
    assert torch.allclose(y, _ref(x, dw, pw, None, None), atol=1e-5)


def check_channel_isolation_of_depthwise(solve):
    # depthwise 阶段通道间不混合：把 pw 设为只取第 1 通道，扰动 x 的第 0 通道不该改变输出
    torch.manual_seed(2)
    x = torch.randn(1, 3, 5, 5)
    dw = torch.randn(3, 1, 3, 3)
    pw = torch.zeros(1, 3, 1, 1)
    pw[0, 1] = 1.0                     # 仅使用通道 1
    y = solve(x, dw, pw)
    x2 = x.clone()
    x2[:, 0] += 50.0
    y2 = solve(x2, dw, pw)
    assert torch.allclose(y, y2, atol=1e-6), "depthwise 应保持通道隔离"


check_basic.example = True
