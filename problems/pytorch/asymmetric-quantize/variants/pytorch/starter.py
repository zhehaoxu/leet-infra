import torch


def solve(x):
    # TODO: 非对称 fake-quant（uint8）。scale=(max-min)/255；
    #   zp = clamp(round(-min/scale), 0, 255)；q = clamp(round(x/scale)+zp, 0, 255)；
    #   返回 (q-zp)*scale。scale==0（常数张量）原样返回。用 torch.round。
    raise NotImplementedError
