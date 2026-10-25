import torch


def solve(A, N, B, D, td, tv):
    """Sequoia 深度受限最优草稿树 DP（arXiv:2402.12374，177 题 T[n,b] 的深度扩展 + 代价目标）。

    A: float 张量，形状 (D, B)。A[d-1][b-1] = 在「父链已全部接受」的前提下，位于**绝对深度 d**
       （根为深度 0）且是父节点**第 b 个孩子**（b=1..B）的节点被接受的概率。
    N: int，树规模预算上限；B: int，每个节点最多孩子数（= A 的列数）；D: int，最大深度（= A 的行数，
       允许 D=0，此时 A 形状 (0, B)，只有根可行）。
    td: float，每草拟一个节点的 draft 代价；tv: float>0，一次 target 验证的固定代价。

    节点分数 f(root)=1，f(v)=∏_{沿根到 v 的路径} A[深度-1][孩子序号-1]；一棵树的期望产出
    F(T)=Σ_{v∈T} f(v)（一次迭代期望吐出的 token 数，含 target 自己的 1 个）。
    求 F(n)=max{F(T): |T|=n, 出度<=B, 深度<=D}，再返回规模在 1..N 上的最优**每秒产出**
        max_n F(n) / (tv + td * n)。

    返回二元组 `(ratio, n_star)`：ratio 为 float 最优目标值，n_star 为达到最优的树规模（int，
    并列时取最小的 n）。不可行的 n（如 n > 1+B+...+B^D）不参与取最大；n=1 恒可行。
    """
    # TODO
    raise NotImplementedError
