import torch


def solve(W, Xc, alphas, wsc_choices, wsc_w, n_bits, lwc_steps):
    """OmniQuant 式逐通道等价变换 + 权重裁剪（WSC 初始化）联合搜索，返回一个 dict。

    W          : float32/float64 (n, n)，某一个方阵线性层的权重（out == in == n）
    Xc         : float32/float64 (T, n)，已按列减过均值（center-shifted）的校准激活；
                 本题不再做中心化，直接用
    alphas     : list[float]，非空，每个在 [0, 1]；等价变换的 alpha 网格（beta 恒等于 alpha）
    wsc_choices: list[str]，非空，元素取 {"min", "ma", "ma2", "2", "1"}；WSC 阈值候选表
    wsc_w      : float32/float64，任意形状的逐 block 权重最大绝对值（调用方给定，不重算），
                 摊平后记 w2 = min(w)、w3 = max(|w|)
    n_bits     : int，[2, 16]，对称整数量化位宽，qmax = 2^(bits-1) - 1、qmin = -2^(bits-1)
    lwc_steps  : int，非负；**本题不执行 LWC 的梯度步**，该值不改变任何返回值

    对每个 alpha（外层，按输入顺序）：
      1. s1[i] = max_j |W[i,j]|（逐输出通道）、s2[j] = max_t |Xc[t,j]|（逐输入通道）；
         s = alpha * s1 + (1 - alpha) * s2。
      2. shift = alpha * (s1 - s) / s1（上游 beta 直接复用 alpha）；s1[i] == 0 处 shift[i] = 0。
         若某个 s[i] == 0，倒数无定义，抛 ValueError。
      3. 记 1 为长度 n 的全 1 列向量、v = shift：
         T     = diag(s)   @ (I - outer(1, v) / n)   ->  T[i,j] = s[i] * (delta_ij - v[j] / n)
         T_inv = diag(1/s) @ (I + outer(1, v) / n)   ->  T_inv[i,j] = (delta_ij + v[j] / n) / s[i]
      4. W' = T @ W @ T_inv；norm_comp = max_ij |(T @ T_inv - I)[i,j]|（无穷范数）。
    对每个 choice（内层，按输入顺序）：
      5. 阈值 t：min -> w2，ma -> w3/4，ma2 -> w3/2，2 -> w2/8，1 -> w3/8（标量，对整张权重生
         效）；Wc = clamp(W', -t, t)。
      6. 逐输出行对称量化：scale_r = max_j |Wc[r,j]| / qmax（该行全 0 时 scale_r = 1、码全 0），
         code = clamp(round(Wc / scale_r), qmin, qmax)（就近、.5 取偶），deq = code * scale_r。
      7. loss = sum_{t,j} ((Xc @ W'^T - Xc @ deq^T)[t,j]^2) / (T * n)。

    全程 float64 计算；取 (alpha, choice) 里 loss 最小者，并列取先出现者
    （等价于 min(range(len(losses)), key=losses.get)）。

    返回 dict，键固定为以下 11 个（浮点张量一律 float32，与输入 dtype 无关）：
      best_alpha_index : int，最优 alpha 在 alphas 里的下标
      best_alpha       : float，alphas[best_alpha_index] 的值
      best_wsc_choice  : str，最优的 choice 名
      losses           : float32 (len(alphas) * len(wsc_choices),)，alpha 外层、choice 内层摊平
      s                : float32 (n,)，最优 alpha 下的 s
      shift_vec        : float32 (n,)，最优 alpha 下的 shift
      transform_T      : float32 (n, n)，最优 alpha 下的 T
      transform_T_inv  : float32 (n, n)，最优 alpha 下的 T_inv
      norm_comp        : float，最优 alpha 下 ||T @ T_inv - I||_inf
      clipped_weight   : float32 (n, n)，clamp(T @ W @ T_inv, -t, t)，t 来自 best_wsc_choice
      lwc_bound        : float32 (wsc_w.numel(),)，|wsc_w| * 2 * sigmoid(0.01)（LWC 的初值界，
                         摊平、逐元素），不随 alpha 变化
    抛 ValueError：W 不是方阵 / W 与 Xc 的输入通道数不一致 / W 或 Xc 不是二维 /
    alphas 或 wsc_choices 为空 / alpha 越出 [0, 1] 或非数值 / choice 不在五元集合内 /
    n_bits 越出 [2, 16] / lwc_steps 为负 / wsc_w 为空 / 某个 s[i] 恰为 0。
    """
    # TODO
    raise NotImplementedError
