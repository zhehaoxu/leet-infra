"""SliceGPT 旋转-切边步 -- 待实现骨架.

回源: microsoft/TransformerCompression ``src/slicegpt/rotate.py``
(SliceGPT, arXiv 2401.15024)。

单层 rotate-and-slice:
1. pca(H = Σ xᵀx, mask==0 先置零, 对角 damp = 0.01·mean(diag(H)),
   eigh 后按特征值降序排列向量) 分别作用于三组显式信号, 得 Q_in/Q_mid/Q_out。
2. W_qkv ← W_qkv @ Q_in (float64 旋转后转回 float32), 切列 [:, :attn_in];
   W_o ← Q_mid.T @ W_o 切行 (o_bias 同理); W_1 ← W_1 @ Q_mid 切列;
   W_2 ← Q_out.T @ W_2 切行。
3. shortcut 链: attn_shortcut = Q_in.T[:attn_in] @ Q_mid[:, :attn_out];
   mlp_shortcut = (Q_mid.T[:attn_out] @ Q_out)[:, :mlp_out]
   (两段 matmul 均为 float32, Q 先 .to(float32))。

请实现 solve:
* 校验三组信号 list / 掩码对齐与 0/1 / 权重形状 / 三个切边维度;
* 返回 dict(w_qkv, w_o, o_bias, w1, w2, attn_shortcut_q, mlp_shortcut_q,
  q_in, q_mid, q_out, eig_vals); 纯函数式, 不修改入参。
"""

import torch


def solve(layer_inputs, mlp_ln_inputs, layer_outputs,
          w_qkv, w_o, w1, w2,
          attn_in_dim, attn_out_dim, mlp_out_dim,
          o_bias=None, ignore_masks=None):
    raise NotImplementedError
