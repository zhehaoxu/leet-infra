import torch


def solve(vectors, dictionary, n_atoms, tol=None):
    """Lexico 的 KV 缓存稀疏字典编码：逐行 OMP（正交匹配追踪），输出 CSR 三元组。

    vectors    : float32/float64 (N, D)，每行是一个 KV 向量（待编码的信号 y）
    dictionary : 与 vectors 同 dtype 的 (D, M)，原子是「列」，第 m 个原子是 dictionary[:, m]
    n_atoms    : 正整数且 <= M，每行的稀疏度预算（Lexico 的 max_sparsity，默认 14）
    tol        : None，或落在 [0, 1) 的浮点数，相对 L2 残差阈值

    逐行独立跑经典贪心 OMP（内部一律 float64，返回时按输入 dtype 转回；不就地改输入）：
      1. S=[]，残差 r=y（初值 0 拟合），initial=‖y‖²。
      2. k=0..n_atoms-1，每轮：
         a. 先做停止判定：若 tol 不为 None 且 ‖r‖² <= tol²·initial，本行到此为止，
            lengths=k（已选原子的个数），本轮及之后不再选任何原子；预算跑满则 lengths=n_atoms。
         b. j = argmax_m |r · dictionary[:, m]|，只在尚未选过的原子里取，同分取最小下标。
         c. 把 j 追加进 S，然后对**整个已选集合**重解最小二乘 w=(X_SᵀX_S)⁻¹X_Sᵀy
            （不是只更新新原子的那一项，也不是增量投影），r = y - X_S w。

    y=0 的零行是确定性的：所有相关系数都是 0，给了 tol 时第 0 轮判定 0 <= 0 直接停，
    lengths=0；不给 tol 时依次选中 0,1,…,n_atoms-1，权重全为 0。

    返回 dict，键固定为 6 个（CSR 打包同上游 omp.py）：
      lengths       : int64 (N,)，每行真正用到的原子数
      csr_indptr    : int64 (N+1,)，[0] + lengths 的前缀和
      csr_indices   : int64 (total,)，各行选中的原子 id 按行拼接，行内是**被选中的先后顺序**
      csr_values    : 输入 dtype (total,)，与 csr_indices 一一对应的最小二乘权重
      residual_norm : 输入 dtype (N,)，该行在自己的停止点上的 ‖y - recon‖
      recon         : 输入 dtype (N, D)，X_S w；lengths==0 的行是全零

    非法输入抛 ValueError：vectors 非二维；dictionary 非二维；dictionary.shape[0] 不等于
    vectors.shape[1];两者之一不是 float32/float64；两者 dtype 不一致；n_atoms 不是正整数
    （bool、3.0 之类都算不合法）或大于 M；tol 既不是 None 也不是 [0,1) 内的浮点数
    （tol=1.0、tol=-0.1、tol=True 都要拒）。
    """
    # TODO
    raise NotImplementedError
