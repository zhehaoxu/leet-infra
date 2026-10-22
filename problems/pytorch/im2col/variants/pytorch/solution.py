"""参考答案：im2col 卷积 —— 把每个感受野展开成列，卷积退化为一次矩阵乘。"""

import torch
import torch.nn.functional as F


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
          stride=1, padding: int = 0) -> torch.Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    sh, sw = stride
    ph, pw = padding
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph), value=0.0)
    n, cin, h, w = x.shape
    cout, _, kh, kw = weight.shape
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    # im2col：每个输出位置的感受野拉平成一行 -> (N, oh*ow, cin*kh*kw)
    cols = x.unfold(2, kh, sh).unfold(3, kw, sw)              # (N,C,nh,nw,kh,kw)
    cols = cols.permute(0, 2, 3, 1, 4, 5).reshape(n, oh * ow, cin * kh * kw)
    # 权重矩阵化：(cin*kh*kw, cout)，一次 matmul 完成全部通道的累加
    rows = weight.reshape(cout, cin * kh * kw).T
    out = cols @ rows                                          # (N, oh*ow, cout)
    out = out.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
