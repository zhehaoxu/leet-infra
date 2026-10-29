import torch


def solve(x, ord=None, dim=None, keepdim=False):
    """vector-norm：torch.linalg.vector_norm 的等价实现（沿一或多根轴把张量压成范数）。

    入参：
    - x: float32 张量，形状任意（内部用 float64 累计，返回 float32）。
    - ord: None / int / float / "inf" / "-inf"，默认 None 即 2-范数。取值语义：
      * None 或 2：sqrt(Σ|x|^2)。
      * 1：Σ|x|。注意 -1 不是绝对值和，而是 (Σ|x|^{−1})^{−1}（只要段内有一个 0
        元素，结果就是 0），与矩阵范数的 -1 口径完全不同。
      * float("inf") 或 "inf"：max|x|；float("-inf") 或 "-inf"：min|x|。
      * 0：非零元素个数（计数，不做幂次）。
      * 其他有限实数 p：(Σ|x|^p)^(1/p)，须用「先提出最大绝对值 m，再算
        m·(Σ(|x|/m)^p)^(1/p)」的稳定式，避免大数上溢 / 小数下溢；全零向量直接返回 0。
      * "fro" / "nuc" 等矩阵专属记号对向量无意义，抛 ValueError。
      注：torch.linalg.vector_norm 的 ord 默认是 2 且只接受 Number；本题接口额外接受
      None（等价于 2）与 "inf" / "-inf" 字符串写法，判题只用本节声明的取值。
    - dim: None / int / 二元以上的 int 元组，默认 None 表示把整个张量展平后取范数（返回 0 维）。
      负数按 x.dim() 折算；越界或重复轴抛 ValueError。
    - keepdim: 是否保留被归约的轴（长度 1）。

    返回与输入同 dtype 的张量；dim=None 且 keepdim=False 时是 0 维张量。
    """
    # TODO
    raise NotImplementedError
