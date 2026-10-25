import torch


def solve(Q, P, tau, L):
    """LayerSkip 自推测解码：逐层早退草稿 + 剩余层验证的调度与代价核算。

    Q:    float (R, g, D, V)。Q[r][j][l] 是请求 r 的第 j 个草稿位（0 起）在**跑完第 l+1 个
          transformer block 后**、用共享 LM head 反嵌入得到的分布（每行和为 1，D <= L）。
    P:    float (R, g, V)。P[r][j] 是完整模型（全部 L 层）在同一个位置上给出的目标分布，
          用来判定草稿位 j 是否被接受。
    tau:  float，早退置信阈值。
    L:    int，模型总 block 数（1 <= D <= L）。

    调度规则（每个请求独立，一轮 = g 个草稿位 + 一次并行验证）：
      1) 草稿位 j 从第 1 个 block 开始逐层加深，取**第一个**满足
         max_v Q[r][j][l][v] >= tau 的层号 l，早退深度 e_j = l + 1；若 D 层都不满足则
         e_j = D。草稿 token d_j = argmax_v Q[r][j][e_j-1][v]（并列取最小下标）。
         草稿阶段代价 = sum_j e_j 个 block。
      2) 验证阶段是一次并行前向，只需跑「草稿缓存没覆盖到的那些层」：
         代价 = max(L - min_j e_j, 1) 个 block（min_j e_j 是所有草稿位共同拥有的缓存前缀；
         若已在最后一层早退，仍要为收尾 token 跑一次 LM head，按 1 个 block 计入）。
      3) 接受判定用**贪心一致**（LayerSkip 原文的 match）：记 t_j = argmax_v P[r][j][v]
         （并列取最小下标），从 j = 0 起，只要 d_j == t_j 就接受；第一次不一致即停止，
         该位置改由目标分布的 argmax 顶上。
      4) 本轮产出 token 数 = 接受的草稿位数 a + 1（a < g 时是 disagreement 处的目标 token，
         a == g 时是全模型多算出的那一个 bonus token）。

    返回四元组：
      toks:  long (R,)，每个请求本轮产出的 token 数（= a + 1）；
      lays:  long (R,)，每个请求本轮实际执行的 block 数；
      depth: long (R, g)，每个草稿位的早退深度 e_j（g == 0 时形状 (R, 0)）；
      speedup: float，整体加速比 = sum(toks) * L / sum(lays)。
      即「同样的 token 数，完整模型逐 token 解码要花的 block 数」÷「实际花的 block 数」；
      注意它是 batch 级的 token 加权口径，不等于各请求 speedup 的算术平均。

    g == 0 表示不做投机（退化为逐 token 全模型解码）：toks = 1，lays = L，depth 为空行，
    此时该请求的 speedup 恰为 1。
    """
    # TODO
    raise NotImplementedError
