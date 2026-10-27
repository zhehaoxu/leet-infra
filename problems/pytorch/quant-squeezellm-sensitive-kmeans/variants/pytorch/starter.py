import torch


def solve(w, sens, k, iters):
    """SqueezeLLM 敏感度加权 1D Lloyd：位置分位初始化 + 加权均值更新。
    w,sens:(N,) float32；1<=k<=N；iters>=0；sens 非负且 sum>0。
    分配：argmin|w-c|（并列取小簇号）；更新：c_j=sum(s*w)/sum(s)（空簇保留）；
    每轮后升序排序。返回 (centroids:(k,) float32 升序, assign:(N,) int64, deq:(N,) float32)。
    非法入参抛 ValueError。"""
    # TODO
    raise NotImplementedError
