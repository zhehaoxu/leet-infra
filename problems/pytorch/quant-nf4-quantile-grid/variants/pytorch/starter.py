import torch


def solve(w, offset):
    """NF4 分位码本构造 + 64 宽块编码。
    w:(M,N) float32 且 N%64==0；offset 在 (0.5,1)。
    码本：正 8 点 p=offset+(0.5-offset)*i/8 (i=0..7)、负 7 点 p=offset+(0.5-offset)*i/7 (i=0..6)，
    取 Phi^-1 后补 0、排序、除以 max|v| 归一化到 [-1,1]；
    每 64 列一块，s=块 absmax，code=argmin|w/s - 码本|（并列取小码），deq=码本[code]*s。
    返回 (codebook:(16,) float32, codes:(M,N) int64, deq:(M,N) float32)。
    offset 非法或 N%64!=0 抛 ValueError。禁止 bitsandbytes / quantile。"""
    # TODO
    raise NotImplementedError
