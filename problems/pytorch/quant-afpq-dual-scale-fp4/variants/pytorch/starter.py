import torch


def solve(W, bits):
    """AFPQ：正负两侧各用一个浮点 scale，在 FP4/FP3 码表上做最近码取整。

    `W` 是 [G, N] 权重，逐行（输出通道）独立处理。码表与最大码由 `bits` 决定：
    - bits=4：maxq=12，half=maxq/2=6，码表 [-6, -4, -3, -2, -1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 6]；
    - bits=3：maxq=8，half=4，码表 [-4, -2, -1, -0, 0, 1, 2, 4]。

    1. 每行 xmin=min(row.min(), 0)，xmax=max(row.max(), 0)；
       若 xmin==0 且 xmax==0，令 xmin=-1、xmax=+1。
    2. scale_pos=xmax/half，scale_neg=|xmin|/half。
    3. 把元素拆成正部 x_pos=max(w,0) 与负部 x_neg=min(w,0)：
       q_pos=x_pos/scale_pos，q_neg=x_neg/scale_neg，各自取码表中距离最近的码
       （距离相同取码表里更靠前的码）。
    4. 重建 dequant = q_pos*scale_pos + q_neg*scale_neg。

    某一侧 scale 为 0 时（整行没有正数或没有负数），该侧的 q 与贡献都取 0，
    不要做 0/0 除法。

    返回 dict：scale_pos/scale_neg/q_pos/q_neg/dequant，均为 float32，形状 [G] 或 [G,N]。
    非法输入（bits 不是 3/4、W 非二维浮点、空张量、非有限值）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
