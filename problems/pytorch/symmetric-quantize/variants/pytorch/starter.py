import torch


def solve(x):
    # TODO: 对称 fake-quant。scale = max|x|/127；q = clamp(round(x/scale), -127, 127)；
    #   返回 q*scale（反量化浮点）。全零张量原样返回。用 torch.round（就近取整）。
    raise NotImplementedError
