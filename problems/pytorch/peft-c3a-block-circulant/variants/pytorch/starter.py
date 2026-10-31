import torch


def solve(x, W0, kernel, b0=None):
    """C3A 块循环卷积适配器（对齐 peft `tuners/c3a/layer.py` 的 get_delta_weight + forward）。
    x:(...,d_in)  W0:(d_out,d_in)  b0:(d_out,) 或 None
    kernel:(d_out//bs, d_in//bs, bs)，bs = kernel.shape[-1]，d_in 与 d_out 都必须能被 bs 整除。
    增量按 bs×bs 分块，第 (m,n) 块由 kernel[m,n] 张成循环矩阵：
        delta[m*bs + i, n*bs + j] = kernel[m, n, (j - i) % bs] / d_in
    即块内每一行都是 kernel[m,n] 循环左移 i 位；最后由 d_in 归一。
    返回 (y, delta)：y = x @ W0.T + b0 + x @ delta.T。
    内部 float64、返回 float32。禁止调用现成 PEFT 库。"""
    # TODO
    raise NotImplementedError
