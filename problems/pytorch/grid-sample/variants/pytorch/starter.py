import torch


def solve(x, grid):
    # TODO: 双线性 grid_sample（mode='bilinear', padding_mode='zeros', align_corners=False）。
    # x:(N,C,H,W), grid:(N,Ho,Wo,2)，[...,0]=x(宽)、[...,1]=y(高)，值域[-1,1]。
    # 像素坐标: x_pix=0.5*(gx+1)*W-0.5, y_pix=0.5*(gy+1)*H-0.5；取四周4像素双线性加权，越界补0。
    raise NotImplementedError
