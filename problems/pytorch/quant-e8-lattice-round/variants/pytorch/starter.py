import torch


def solve(X):
    """E8 格点最近点取整。

    E8 可写成两个陪集的并：D8（坐标全为整数且坐标和为偶数）与 D8+1/2
    （坐标全为半整数，且整数部分之和为偶数）。

    对每个 8 维点：
    1. 在 D8 中找最近点：先逐坐标四舍五入得到 y；若 sum(y) 为偶数则 y 已在 D8；
       否则把 |x_i - y_i| 最大的那个坐标朝 x 的方向移动 1（并列取最小下标，
       差值为 0 时约定 +1），得到坐标和为偶数的 y。
    2. 在 D8+1/2 中找最近点：令 z = x - 0.5，对 z 做同样的 D8 取整，再加回 0.5。
    3. 比较两个候选的平方误差；coset 1 严格更小才选 1，平局选 coset 0。

    输入 X 为 float32 [N, 8]；返回 dict：
    - rounded: float32 [N, 8]，nearest E8 点；
    - coset: int64 [N]，0 表示 D8、1 表示 D8+1/2；
    - sse: float32 [N]，每个点的平方误差。
    内部用 float64，输出 float32。X 非二维浮点、形状不是 [N,8] 且 N>0、
    或含非有限值时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
