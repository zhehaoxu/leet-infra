"""VITS 归一化流前向/反向与 log-det 账 —— 骨架。

你需要实现四个 mode，由 solve(mode, **kw) 分派：

  flow_primitive(kind, shape, x, mask, m=None, logs=None, reverse=False)
      → {"y": [...], "logdet": [...]}

  coupling_layer(shape, x, mask, m, logs=None, mean_only=True, reverse=False)
      → {"y": [...], "logdet": [...]}

  coupling_block(shape, x, mask, m_list, reverse=False)
      → {"y": [...], "logdet": [...]}

  duration_nll(w, e_q, z_u, z1, mask, post_logdets, flow_logdets)
      → {"logq", "nll", "total", "logdet_tot_q", "logdet_tot", "z0", "z0_log"}

关键口径：

- 三维张量的形状统一是 (B, C, T)；x 按此形状展平后参与计算。
- mask 官方叫 x_mask：coupling/block 用 (B, 1, T)（对 channel 广播），
  Log / ElementwiseAffine 要求与 x 同形 (B, C, T)。
  本题两种都接受，内部统一归一化到 (B, C, T)。
- m / logs 是逐 channel 的参数：标量（广播）、长度 C 的一维列表、
  或 (C, 1) 的嵌套列表。
- 所有非法输入一律抛 ValueError。
- 能走精确有理数的部分走 fractions.Fraction，log / exp 按 float 算；
  最终出参统一转 float。

四个容易写错的点：

1. Flip 翻的是 channel 轴（torch.flip(x, [1])），而且**完全忽略 mask**。
2. coupling 前向是 `x1 = m + x1 * exp(logs) * mask`：
   mask 只乘在 x1 项上，m 不被 mask → mask=0 的位置输出是 m。
3. coupling 的 logdet 是 sum(logs)，**不乘 mask**；
   ElementwiseAffine 的 logdet 是 sum(logs * mask)，**要乘 mask**。
4. logq = gauss - (post_logdets + lsig)：logsigmoid 那一项是**被减去的**。

具体出参格式见题面。
"""

from fractions import Fraction


def solve(mode, **kw):
    raise NotImplementedError("请实现 solve(mode, **kw)")
