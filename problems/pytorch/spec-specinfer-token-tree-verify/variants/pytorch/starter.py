import torch


def solve(parent, token, qdist, T, u, u_final):
    """SpecInfer 通用 token-tree 并行验证（arXiv:2305.09781，tree-attention 一次前向验证整棵草稿树）。

    parent: long (M,)。节点 0 是已确认 token（根，parent[0] = -1），其余 parent[i] < i（拓扑序）。
    token:  long (M,)。每个草稿节点提出的 token id（根的不使用），取值在 [0, V)。
    qdist:  float (M, V)。draft 模型在**该节点位置上**用来抽样 token 的分布；
            q_v = qdist[v][token[v]] 即该草稿 token 的 draft 概率（根的行的内容不使用）。
    T:      float (M, V)。target 模型在 tree-attention 一次前向后**输出在节点 v 上**的分布，
            它是 v 的**孩子位置**上的目标分布：孩子 c 的 p_c = T[parent[c]][token[c]]。
            （根的 row 给出第一层孩子的分布；叶子孩子的分布不存在，故收尾用父节点自己的 row。）
    u:      float (M,)。逐节点的显式均匀数（接受硬币），u[0] 不使用，取值在 [0, 1)。
    u_final: float，收尾 token 采样用的均匀数，取值在 [0, 1)。

    验证规则（沿接受前缀逐层推进）：
      accepted = {root}, frontier = [root]；反复：
        候选 = frontier 各节点的全部孩子（按节点编号升序）；候选为空 -> 树耗尽，停止；
        对每个候选 v（父为 par）：ratio = p_v/q_v = T[par][token[v]] / qdist[v][token[v]]，
          若 q_v == 0 视为 ratio = +inf；当 u[v] < min(1, ratio) 时接受 v；
        本层无任何节点被接受 -> 停止（这些孩子全部被拒）；否则 accepted 并入本层存活者，
          frontier = 本层存活者。
      取**最深**的已接受节点 v*（并列时取 token 序列字典序最小者，再并列取节点编号最小者），
      root->v* 的 token 序列就是存活路径（不含根）。
      收尾 token：若 v* 有被拒的孩子（即 v* 在本层全军覆没的那一层里挂过候选），
        取编号最小的被拒孩子 c，用残差分布 norm(max(T[v*] - qdist[c], 0)) 采样（逐元素先取
        非负、再除以元素和）；否则（树耗尽 / v* 是叶子）直接从 T[v*] 采样 bonus。
      采样 = CDF 反查：返回**最小**的 j 使 cumsum_j > u。

    返回二元组 `(out, nodes)`，都是一维 long 张量：
      out   = 存活路径的 token id + 收尾 token，长度 = len(nodes) + 1（恒 >= 1）；
      nodes = 存活路径的节点索引（升序即根到 v* 的次序，不含根，不含收尾 token）。
    """
    # TODO
    raise NotImplementedError
