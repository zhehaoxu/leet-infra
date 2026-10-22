"""参考答案：二维卷积前向 —— 输出位循环 + 逐输入通道切片点乘求和（朴素教学实现）。"""

import torch


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def solve(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
          stride=1, padding: int = 0) -> torch.Tensor:
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    if ph or pw:
        x = torch.nn.functional.pad(x, (pw, pw, ph, ph), value=0.0)
    n, cin, h, w = x.shape
    cout, cin_k, kh, kw = weight.shape
    assert cin_k == cin, "权重输入通道数必须与输入一致"
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    out = x.new_zeros((n, cout, oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = x[:, :, i * sh: i * sh + kh, j * sw: j * sw + kw]  # (N,Cin,kh,kw)
            # 逐输出通道：与 weight[o] 对应元素相乘后全和（等价于逐通道相关）
            for o in range(cout):
                out[:, o, i, j] = (region * weight[o]).sum(dim=(1, 2, 3))
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
