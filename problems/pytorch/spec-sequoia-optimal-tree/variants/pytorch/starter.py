def solve(N, B, P):
    """Sequoia 最优草稿树（Algorithm 1）：位置接受率 P[b-1]=根的第 b 个孩子被接受概率，
    节点分数 f(v)=∏_{i∈Path(v)} P[i]、F(T)=Σ_{v∈T} f(v)。用二维 DP T[n,b]（size n、根
    FirstBranch=b）求大小为 N、每点出度≤B 的树的最大期望生成 token 数。返回 float=Tmax[N]。"""
    # TODO
    raise NotImplementedError
