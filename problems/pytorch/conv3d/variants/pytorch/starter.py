import torch


def solve(x, weight, bias, stride=1, padding=0):
    # TODO: 三维卷积（互相关）。x:(N,Cin,D,H,W), weight:(Cout,Cin,kD,kH,kW), bias:(Cout,) 或 None。
    # 先在三个空间维两侧各补 padding 个零，各维输出长度 L_out=(L+2*padding-kL)//stride+1。
    #   out[n,o,a,b,c] = bias[o] + sum_ci sum_di sum_hi sum_wi
    #                       x[n,ci,a*s+di,b*s+hi,c*s+wi] * weight[o,ci,di,hi,wi]
    # 结果须与 F.conv3d(x, weight, bias, stride=stride, padding=padding) 一致。
    raise NotImplementedError
