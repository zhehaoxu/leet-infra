import torch


def solve(params, decay=0.9, warmup=True):
    # TODO: 返回一个 EMA 对象，暴露 .shadow（与 params 一一对应、构造时=p 的 clone）与
    #   .update()（就地：先 n←n+1，再 d = min(decay,(1+n)/(10+n)) if warmup else decay，
    #   对每个参数 shadow ← d*shadow + (1-d)*param）。update() 不修改 params、影子与 param 分离。
    raise NotImplementedError
