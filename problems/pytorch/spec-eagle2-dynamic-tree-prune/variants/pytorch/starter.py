import torch


def solve(parent, acc, k, gmax):
    """EAGLE-2 动态草稿树重排/剪枝（arXiv:2406.16858，Algorithm 1）。

    parent: long 张量，形状 (M,)。节点 0 是已确认 token（根，parent[0] = -1），其余节点满足
        parent[i] < i（拓扑序），即一棵静态草稿树：Medusa/EAGLE 第一遍草拟出来的父子结构。
    acc: float 张量，形状 (M,)。acc[i] = EAGLE-2 第二遍前向估出的**条件接受率**——在 i 的父节点
        已被接受的前提下，节点 i 的 token 被 target 接受的概率；acc[0] 不被使用。
    k: int，本轮送入 target 并行验证的**草稿节点数预算**（根槽位免费，不占预算）。
    gmax: int，最多草拟的层数（树高上限），gmax >= 0。

    累积分数（EAGLE-2 的 confidence）：V(root)=1，V(v)=∏_{u 在根到 v 的路径上, u != root} acc[u]，
    沿路径单调不增。动态选点（逐层贪心，与论文 Algorithm 1 一致）：
      frontier = [root]；used = 0；level = 0
      当 used < k 且 level < gmax 且 frontier 非空时：
        候选 = frontier 中每个节点的全部孩子（按节点编号升序收集）；
        候选为空则停止；
        把候选按 (-V(v), 节点编号) 升序排序，取前 min(剩余预算, 候选数) 个并入已选集合，
        frontier = 这一批新节点，level += 1。
    注意预算是**逐层填满**的：某一层的候选少于剩余预算时，本层全留并把余量交给下一层，
    这与「对全部节点取全局 top-k」在稀疏层处会给出不同的点集（见 notes）。

    返回二元组 `(sel, E)`：
      sel: 1 维 long 张量，被选中送入 target 验证的节点索引，**升序**排列（不含根；长度 <= k）；
           该集合对祖先封闭——任何被选节点的父节点（除非是根）也在集合内。
      E: float，本轮迭代的期望吐出 token 数 = 1 + Σ_{v∈sel} V(v)，其中 1 是 target 自身必得的那
         一个 token（EAGLE-2 Lemma 1 的等价式：必得首 token + 逐节点累积接受率之和）。
    """
    # TODO
    raise NotImplementedError
