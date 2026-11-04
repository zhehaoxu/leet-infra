"""SliceGPT 旋转-切边步.

回源: microsoft/TransformerCompression ``src/slicegpt/rotate.py`` (528 行,
SliceGPT 论文 arXiv 2401.15024) 的 ``pca_calc`` (L501-528) 与
``rotate_and_slice_sequential`` (L140-251)、``slicing_scheduler.py`` 的
ForwardSlicingScheduler 维度一致性 (L176-196)。

SliceGPT 核心: 用校准信号的相关矩阵 H = Σ xᵀx 的特征向量做正交旋转 Q,
把隐空间旋到「主成分坐标」后直接**切掉尾部行** (低能量溶剂空间)——
权重不置零、不量化, 而是整体降秩 d → d'。旋转可逆性由相邻块间的
shortcut 旋转矩阵 (attn_shortcut_Q / mlp_shortcut_Q) 链式拼接维持。

单层 rotate-and-slice 流水线 (官方顺序块, 信号已显式入参):
    1. Q_in  = pca(layer_inputs)         # 上一块输出空间的主成分
       attn_shortcut = Q_in.T (float32);  W_q,k,v ← W_qkv @ Q_in, 切列 [:, :attn_in]
       attn_shortcut ← attn_shortcut[:attn_in, :]
    2. Q_mid = pca(mlp_ln_inputs)        # 注意力与 MLP 之间的 LayerNorm 输出
       attn_shortcut ← attn_shortcut @ Q_mid[:, :attn_out]   # float32 matmul
       W_o ← Q_mid.T @ W_o, 切行 [:attn_out, :];  b_o 同理切行
       mlp_shortcut = Q_mid.T[:attn_out, :]                  # float32
       W_1 ← W_1 @ Q_mid, 切列 [:, :attn_out]
    3. Q_out = pca(layer_outputs)        # 本块输出 (给下一块做 Q_in)
       mlp_shortcut ← mlp_shortcut @ Q_out, 再切列 [:, :mlp_out]
       W_2 ← Q_out.T @ W_2, 切行 [:mlp_out, :]

pca (官方 pca_calc):
    H = Σ_batch Σ_token x xᵀ  (mask==0 的 token 先置零)
    damp = 0.01 · mean(diag(H));  H 对角加 damp      # 防退化
    torch.linalg.eigh(H) (升序) → 按特征值**降序**重排列向量 → Q 列即主成分

与官方实现的判题域约定 (official -> judgement):
* 官方在真实模型上前向得到块间信号; 判题域 layer_inputs / mlp_ln_inputs /
  layer_outputs 三组信号全部显式入参 (每组是 list[(B, S, d)], 逐 batch 累积 H)。
* 切边维度由 scheduler 提供; 判题域显式入参 attn_in_dim / attn_out_dim /
  mlp_out_dim (顺序块一致性: mlp 输入维 ≡ attn 输出维)。
* 权重旋转在 float64 中做再转回 float32; shortcut 拼接按官方用 float32
  (Q 先 .to(float32) 再 matmul)。final_orientation='pca' (random 分支不入)。
* 官方 pca 就地掩码; 判题域纯函数式——掩码置零前 clone, 不动入参。
* 特征分解约定: 必须 torch.linalg.eigh(同环境同后端, 符号/重排序确定),
  测试谱非退化 (特征值两两可分)。
* 官方不校验; 判题域补信号组/掩码/权重形状/切边维度等显式校验。
"""

import torch


def _pca(batches, masks):
    """官方 pca_calc: 掩码置零 → H = Σ xᵀx → 对角 damping → eigh 降序."""
    H = None
    for idx, x_batch in enumerate(batches):
        x = x_batch.clone() if masks is not None else x_batch
        if masks is not None:
            x[masks[idx] == 0] = 0
        x = x.double()
        H_batch = torch.sum(x.mT @ x, dim=0)
        H = H_batch if H is None else H + H_batch
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1])
    H[diag, diag] = H[diag, diag] + damp
    eig_val, eig_vec = torch.linalg.eigh(H)
    index = torch.argsort(eig_val, descending=True)
    return eig_val[index], eig_vec[:, index]


def _check_signals(signals, masks, name):
    if not isinstance(signals, list) or not signals:
        raise ValueError(f"{name} 必须是非空 list")
    for i, sig in enumerate(signals):
        if not isinstance(sig, torch.Tensor):
            raise ValueError(f"{name}[{i}] 必须是 torch 张量")
        if sig.dim() != 3:
            raise ValueError(f"{name}[{i}] 必须是 3 维 (B, S, d) 张量")
        if sig.shape[-1] != signals[0].shape[-1]:
            raise ValueError(f"{name}[{i}] 的隐维必须一致")
        if masks is not None:
            if not isinstance(masks[i], torch.Tensor) or masks[i].shape != sig.shape[:2]:
                raise ValueError(f"ignore_masks[{i}] 必须与 {name}[{i}] 的 (B, S) 对齐")
            if not torch.all((masks[i] == 0) | (masks[i] == 1)):
                raise ValueError(f"ignore_masks[{i}] 必须是 0/1 掩码")


def _check_dim(dim, d, name):
    if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= d:
        raise ValueError(f"{name} 必须是 [1, {d}] 内的整数")


def solve(layer_inputs, mlp_ln_inputs, layer_outputs,
          w_qkv, w_o, w1, w2,
          attn_in_dim, attn_out_dim, mlp_out_dim,
          o_bias=None, ignore_masks=None):
    """SliceGPT 单层旋转-切边步。

    参数
    ----
    layer_inputs : list[torch.Tensor]
        进入本块的信号 (上一块输出空间), 每个 (B, S, d); 决定 Q_in。
    mlp_ln_inputs : list[torch.Tensor]
        注意力与 MLP 之间的信号 (LayerNorm 输出), 决定 Q_mid。
    layer_outputs : list[torch.Tensor]
        本块输出信号 (MLP 输出旋转/切边之前的 d 维空间), 决定 Q_out。
    w_qkv : list[torch.Tensor]
        注意力输入权重 (W_q/W_k/W_v), 每个 (d, d), 行主序 out×in。
    w_o : torch.Tensor
        注意力输出权重 (d, d)。o_bias : 可选 (d,) 偏置。
    w1, w2 : torch.Tensor
        MLP 升维 (f, d) 与降维 (d, f) 权重。
    attn_in_dim / attn_out_dim / mlp_out_dim : int
        切边后的维度 (1 <= dim <= d); attn_out_dim 同时是 MLP 输入维。
    ignore_masks : list[torch.Tensor] or None
        每组信号的 (B, S) 0/1 掩码, 0 的 token 在 H 累积前置零。

    返回
    ----
    dict(w_qkv, w_o, o_bias, w1, w2, attn_shortcut_q, mlp_shortcut_q,
         q_in, q_mid, q_out, eig_vals)
        全部权重为旋转+切边后的 float32; shortcut 是 (attn_in, attn_out) 与
        (attn_out, mlp_out) 的链式拼接矩阵; eig_vals 是三组降序特征值。
    """
    if ignore_masks is not None and not isinstance(ignore_masks, list):
        raise ValueError("ignore_masks 必须是 list 或 None")
    _check_signals(layer_inputs, ignore_masks, "layer_inputs")
    _check_signals(mlp_ln_inputs, ignore_masks, "mlp_ln_inputs")
    _check_signals(layer_outputs, ignore_masks, "layer_outputs")
    if len(mlp_ln_inputs) != len(layer_inputs) or len(layer_outputs) != len(layer_inputs):
        raise ValueError("三组信号的 batch 数必须一致")

    d = layer_inputs[0].shape[-1]
    if not isinstance(w_qkv, list) or not w_qkv:
        raise ValueError("w_qkv 必须是非空 list")
    for i, w in enumerate(w_qkv):
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.shape != (d, d):
            raise ValueError(f"w_qkv[{i}] 必须是 ({d}, {d}) 张量")
    for w_name, w, shape in (("w_o", w_o, (d, d)), ("w1", w1, None), ("w2", w2, None)):
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            raise ValueError(f"{w_name} 必须是 2 维张量")
    if w1.shape[1] != d or w2.shape[0] != d:
        raise ValueError("w1 的入维 / w2 的出维必须等于隐维 d")
    if o_bias is not None and (not isinstance(o_bias, torch.Tensor)
                               or o_bias.dim() != 1 or o_bias.shape[0] != d):
        raise ValueError(f"o_bias 必须是长度 {d} 的一维张量或 None")
    _check_dim(attn_in_dim, d, "attn_in_dim")
    _check_dim(attn_out_dim, d, "attn_out_dim")
    _check_dim(mlp_out_dim, d, "mlp_out_dim")

    dtype = w_qkv[0].dtype

    # 1. Q_in: 旋转注意力输入, 切列; shortcut 起始 = Q_in.T (float32), 切行
    eig_in, q_in = _pca(layer_inputs, ignore_masks)
    attn_shortcut = q_in.t().clone().to(dtype)
    w_qkv_out = [(w.double() @ q_in).to(dtype)[:, :attn_in_dim] for w in w_qkv]
    attn_shortcut = attn_shortcut[:attn_in_dim, :]

    # 2. Q_mid: 注意力输出/MLP 输入的旋转与切边, shortcut 第一段拼接
    eig_mid, q_mid = _pca(mlp_ln_inputs, ignore_masks)
    attn_shortcut = attn_shortcut @ q_mid.to(dtype)[:, :attn_out_dim]
    w_o_out = (q_mid.t() @ w_o.double()).to(dtype)[:attn_out_dim, :]
    o_bias_out = None if o_bias is None else \
        (q_mid.t() @ o_bias.double()).to(dtype)[:attn_out_dim]
    mlp_shortcut = q_mid.t().clone().to(dtype)[:attn_out_dim, :]
    w1_out = (w1.double() @ q_mid).to(dtype)[:, :attn_out_dim]

    # 3. Q_out: MLP 输出旋转与切边, shortcut 第二段拼接后切列
    eig_out, q_out = _pca(layer_outputs, ignore_masks)
    mlp_shortcut = mlp_shortcut @ q_out.to(dtype)
    w2_out = (q_out.t() @ w2.double()).to(dtype)[:mlp_out_dim, :]
    mlp_shortcut = mlp_shortcut[:, :mlp_out_dim]

    return {
        "w_qkv": w_qkv_out, "w_o": w_o_out, "o_bias": o_bias_out,
        "w1": w1_out, "w2": w2_out,
        "attn_shortcut_q": attn_shortcut, "mlp_shortcut_q": mlp_shortcut,
        "q_in": q_in, "q_mid": q_mid, "q_out": q_out,
        "eig_vals": [eig_in, eig_mid, eig_out],
    }
