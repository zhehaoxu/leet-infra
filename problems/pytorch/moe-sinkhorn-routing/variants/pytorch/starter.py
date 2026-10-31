import torch


def solve(logits, iters, tau=1.0, square=False, eps=0.0):
    """中文契约 / English contract
    logits [T,E]：CPU float64 路由 logits，T,E >= 1 / CPU float64 routing logits.
    iters：真正的 Python int，0 <= iters <= 16 / a true Python int in [0, 16].
    tau：正的有限 int/float；square：真正的 Python bool；eps：非负有限 int/float /
        tau positive finite; square a true Python bool; eps non-negative finite.
    结构或标量非法时抛 ValueError；不修改输入，接受非连续张量 /
        raise ValueError for anything else; mutate no inputs and accept noncontiguous tensors.
    R = softmax(logits / tau, dim=-1)；每一轮先把每列除以该列在 token 轴上的和，
    再把每行除以该行的和；所有除数都是 sum + eps（eps 加在除数上，不是被除数上）/
        each round divides columns by their token-axis sums, then divides rows by their row sums;
        every divisor is sum + eps.
    iters=0 且 square=False 就是普通行 softmax；eps=0 时死列会算出 0/0 = NaN 并扩散，
    这是契约要求如实保留的行为 / with eps=0 a zero column yields 0/0 = NaN, which must propagate.
    square=True 时在最后一轮之后再平方一次并按行归一化 /
        when square is set, entries are squared once more and rows renormalized.
    返回 / Returns (Q [T,E] float64, col_load [E] float64)，col_load 是返回矩阵的列和 /
        col_load is the column sums of the returned matrix.
    """
    raise NotImplementedError
