def solve(p_cmp_heads, L, l, top_n, window, gates, branch_outputs):
    """NSA（DeepSeek 原生稀疏注意力，arXiv 2502.11089）的三分支调度账。

    背景机制（论文 3.3 节）：
      NSA 把每个 query 位置的 KV 重映射进三条独立分支——压缩（cmp）、
      选择（slc）、滑窗（win）——再由可学习门控 sigmoid 权重加权聚合
      （式 5/6）。本题考察这条"调度账"：GQA 组内共享分数、top-n 块选择、
      各分支 KV 长度与门控聚合。注意力的 softmax/matmul 本体不在此题范围，
      以中间量直接给出。

    简化口径（全部钉死，避免歧义）：
      - 压缩块长 l、压缩步长 d、选择块长 l' 三者相等（l = d = l'），
        此时论文式 (9) 退化为 p_slc = p_cmp 直接复用；候选选择块与压缩块一一对应。
      - 已生成 L 个 token（编号 0..L-1），当前 query 关注全部因果内 token。
      - 压缩块 i 覆盖 token [i*l, (i+1)*l)，只有**整块落在因果内**（
        即 (i+1)*l <= L）的块才存在；可用压缩块数 n_blocks = L // l。
      - 选择只发生在完整块上（式 12 取整块 concat），部分块由滑窗分支兜底。

    参数：
      p_cmp_heads : list[list[float]]，每个 query head 的压缩块注意力分数
                    （式 8 已 softmax 的 p_cmp；所有 head 等长，长度为 n_blocks）。
                    GQA 组内共享 KV，故块选择必须组内一致。
      L           : 已生成 token 数（L >= 1）。
      l           : 块长（= 压缩步长 = 选择块长，l >= 1）。
      top_n       : 选择分支保留的块数（式 11：rank(p[i]) <= n）。
      window      : 滑窗分支的窗口大小 w。
      gates       : [g_cmp, g_slc, g_win]，三分支的门控权重（已过 sigmoid）。
      branch_outputs : [[o_cmp], [o_slc], [o_win]]，三分支的注意力输出向量
                    （三者长度相同，来自各自独立的 KV——论文的防 shortcut 设计）。

    计算规则（逐条对应论文）：
      1. 组内共享分数（式 10）：p_shared[i] = sum_h p_cmp_heads[h][i]
         —— 组内各 query head 的分数**求和**（不是平均、不是最大）。
      2. top-n 选择（式 11）：对 p_shared 按降序排名（rank=1 为最高分），
         rank <= top_n 的块入选。并列分数按**块索引升序**依次排名
         （稳定：分数相同时索引小者 rank 更靠前）。
         实际入选块数 n_sel = min(top_n, n_blocks)。
      3. 三分支 KV 长度（式 6 的 size 项）：
         cmp: n_blocks * l；slc: n_sel * l；win: min(window, L)。
         N_t = 三者之和。
      4. 门控聚合（式 5）：output[j] = g_cmp*o_cmp[j] + g_slc*o_slc[j] + g_win*o_win[j]。

    返回 dict：
      {
        "shared_scores": [float]（长度 n_blocks），
        "selected_blocks": [int]（入选块索引，升序），
        "cmp_kv_len": int,
        "slc_kv_len": int,
        "win_kv_len": int,
        "n_remapped": int,
        "output": [float]（与 branch_outputs 各向量同长）
      }

    输入不合法时（p_cmp_heads 为空、head 长度不一、top_n < 1、window < 1、
    gates 长度不为 3、三分支输出长度不一或为空）抛 ValueError。
    """
    # TODO
    raise NotImplementedError
