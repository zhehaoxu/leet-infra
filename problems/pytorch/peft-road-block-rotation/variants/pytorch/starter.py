import torch


def solve(theta, alpha, variant, group_size, out_features, x):
    """RoAD：把输出向量按 group 切成 2 维对，每对乘一个带缩放的旋转矩阵，返回旋转阵与旋转后的输出。

    theta      : (size,) float32，旋转角（size 随 variant 变化，见下）
    alpha      : (size,) float32，每个位置的缩放
    variant    : "road_1" / "road_2" / "road_4"
    group_size : int，分组大小（正偶数）
    out_features : int，输出维度，必须能被 group_size 整除
    x          : (..., out_features) float32，待旋转的张量（ROAD 作用在基座层的输出上）

    把输出维度切成 out_features/group_size 个 group，每个 group 再切成前后两半：
    group 内第 j 个元素（j < group_size/2）与第 j + group_size/2 个元素配成一对。

    theta/alpha 的长度随 variant：
      road_1 : out_features/2 —— 每个 group 只有 group_size/2 个参数，同一对共用，且两列共用同一个角
      road_2 : out_features   —— 每个位置各一个角
      road_4 : out_features*2 —— 每对用两个角、两个缩放：前半决定第一列（cos 项），后半决定第二列（sin 项）

    记每个位置 k 的 first_col[k]、second_col[k]：
      road_1 : theta/alpha 先 reshape 成 (n_group, group_size/2) 再逐元素 repeat_interleave(2)
               first_col = alpha * cos(theta); second_col = alpha * sin(theta)
      road_2 : first_col = alpha * cos(theta); second_col = alpha * sin(theta)
      road_4 : theta/alpha reshape 成 (n_group, 2, group_size)
               first_col  = alpha[:,0,:] * cos(theta[:,0,:]); second_col = alpha[:,1,:] * sin(theta[:,1,:])
               两者都按 group 展平

    旋转阵 R（out_features x out_features）：
      R = diag(first_col) + 第二列贡献
      第二列贡献：把 second_col 按 (n_group, 2, group_size/2) 交换前后两半并展平，
      取其对角，再 reshape 成 (n_group, 2, group_size/2, out_features) 后交换前后两半，
      且第一半整体取负。

    输出（逐元素形式，等价于 R @ x）：
      x 按 group 切前后两半 x1、x2，rotate_half = stack(-x2, x1)，
      y = x * first_col + rotate_half * second_col

    返回 (R, y)：R:(out_features, out_features) float32；y 与 x 同形 float32。
    """
    # TODO
    raise NotImplementedError
