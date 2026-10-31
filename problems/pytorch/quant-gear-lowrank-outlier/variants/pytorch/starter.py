import torch


def solve(kv, bits, group_size, sparsity, rank, loop, p_base, q_base):
    """GEAR：离群元精确保留 + 分组非对称量化 + 量化残差的低秩幂迭代补偿。

    kv       : float32/float64 (n_batch, n_head, seq_len, head_dim)，上游 GEARLM 的 K/V 布局
    bits     : int >= 1，每组的量化级数是 2**bits - 1 个间隔
    group_size : int >= 1，必须整除 n_head*head_dim（分组沿合并后的通道轴切）
    sparsity : [0,1) 的浮点数，离群元占比
    rank     : int >= 1 且 <= min(seq_len, head_dim)，残差低秩补偿的秩
    loop     : int >= 1，幂迭代轮数
    p_base   : 与 kv 同 dtype 的 (n_batch, n_head, head_dim, rank)，右侧基
    q_base   : 与 kv 同 dtype 的 (n_batch, n_head, seq_len, rank)，左侧基

    口径（内部 float64，返回按 kv 的 dtype；两个基张量只读，不就地改）：
      1. 把 (B,H,L,D) 沿 head/head_dim 合并成 (B, L, C=H*D) 的「token 行」；
         离群名额 n_out = int(int(B*H*L*D*sparsity) / B / L / 2)——两次向下取整，
         且除式里没有 H。
      2. 每行取 n_out 个最小值和 n_out 个最大值（同分取更小的下标），先把小的那批
         换成**整行原始均值**（含离群元），再把大的那批也换成同一个均值。
      3. 分组非对称量化：每个 (b,l) 行按连续 group_size 切组，mn/mx 为组内极值，
         scale=(mx-mn)/(2**bits-1)，code=round(relu((x-mn)/scale))，deq=code*scale+mn；
         scale==0 的常数组直接反量化成 mn。
      4. 把两组离群元的**原始值**写回反量化结果 → quantized；离群位置的 residual 恒为 0。
      5. 低秩补偿：对每个 (b,h) 的残差 A(L,D)，按上游循环
         for i in range(loop): 若 i 是最后一轮先 P=qr(P).Q；Q=A@P；
         若 i 是最后一轮再 Q=qr(Q).Q；P=Aᵀ@Q；low_rank=Q@Pᵀ。
      6. output = quantized + low_rank。

    返回 dict，键固定为 6 个：
      outlier_mask : bool (n_batch, n_head, seq_len, head_dim)
      sparse_filled: 极值被均值替换后的张量
      quantized    : 量化并把原始极值写回后的张量
      residual     : kv - quantized
      low_rank     : 残差的秩-r 幂迭代近似
      output       : quantized + low_rank

    非法输入抛 ValueError。
    """
    # TODO
    raise NotImplementedError
