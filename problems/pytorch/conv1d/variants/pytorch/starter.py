import torch


def solve(x, weight, bias, stride=1, padding=0):
    # TODO: 一维卷积（互相关）。x:(N,Cin,L), weight:(Cout,Cin,kL), bias:(Cout,) 或 None。
    # 先在长度维两侧补 padding 个零，输出长度 L_out=(L+2*padding-kL)//stride+1。
    #   out[n,o,t] = bias[o] + sum_ci sum_j x[n, ci, t*stride+j] * weight[o, ci, j]
    # 结果须与 F.conv1d(x, weight, bias, stride=stride, padding=padding) 一致。
    raise NotImplementedError
