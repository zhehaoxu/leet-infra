def solve(x_init, velocity_fn, n_steps, t_start=1.0, t_end=0.0):
    """对速度场做显式 Euler 积分（Flow Matching / Rectified Flow 的采样器）。

    x_init:起点（t = t_start 处的状态，flow matching 里就是纯噪声 x_1）；
    velocity_fn:callable(x, t) -> 与 x 同形状的速度（张量或标量）；
    n_steps:步数；t_start / t_end:积分的起止时间，要求 t_end < t_start。

    栅格：t_k = t_start + k·(t_end - t_start) / n_steps（k = 0..n_steps，均匀、含两端）；
    更新：x_{k+1} = x_k + (t_{k+1} - t_k) · velocity_fn(x_k, t_k)。

    返回 (trajectory, x_final)：trajectory 是 n_steps+1 个 float32 张量（含起点）。
    """
    # TODO
    raise NotImplementedError
