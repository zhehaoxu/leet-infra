import torch


def solve(x, kernel_size, stride=None, p=2, mode="lppool2d", padding=0,
          output_size=None, indices=None):
    """池化三态：LpPool（1-D/2-D）、带 indices 的 maxpool、maxunpool。

    入参：
    - x: float 张量（CPU，float32 / float64），且不含 NaN / ±inf。
      * "lppool2d" / "maxpool_idx" / "maxunpool"：恰 4 维 (N, C, H, W)；
      * "lppool1d"：恰 3 维 (N, C, L)。秩不符抛 ValueError。
    - kernel_size: int 或 (kh, kw)（1-D 模式下逐维必须一致），各维正整数。
    - stride: int 或 (sh, sw)；None 时默认等于 kernel_size（逐维取同维 kernel）；各维正整数。
    - p: LpPool 的指数，须 p >= 1 且有限；p < 1 或非有限抛 ValueError（仅 lppool 两模式
      使用并校验；maxpool_idx / maxunpool 忽略 p）。输出 = (Σ_窗口 |x_i|^p)^(1/p)，
      注意带 |x|（对负输入良定义；p=1 即窗口 |x| 求和而非均值，kernel=1 时输出 = |x| 逐元素）。
    - padding: int 或 (ph, pw)，默认 0，每边须 <= 对应 kernel 维的一半，否则抛 ValueError。
      "lppool*" 补 0（全零窗口输出 0）；"maxpool_idx" 补 −inf（对齐本库 maxpool2d 题口径）；
      "maxunpool" 要求 padding == 0，否则抛 ValueError。
    - output_size: 仅 "maxunpool" 使用，int 或 (H, W)。None 时按 (h−1)*stride + kernel
      逐维推断；给定值须逐维 >= 小图对应维，否则抛 ValueError。
    - indices: 仅 "maxunpool" 使用（其余模式忽略）——maxpool 产出的索引张量（即
      "maxpool_idx" 的第二个返回值），形状与 x 一致、整数 dtype、取值须落在 [0, H*W)
      （逐通道空间图展平），否则抛 ValueError。maxunpool 需要 indices 输入，故在本签名
      末尾追加该关键字参数。

    四种 mode（其余取值抛 ValueError）：
    - "lppool2d"：x (N,C,H,W) → (N,C,oh,ow)，oh = floor((H+2ph−kh)/sh)+1（须 >= 1，否则
      抛 ValueError），语义同 F.lp_pool2d（floor 模式；p=2 时与原生逐位一致，本文口径
      带 |x|，负输入的奇数 / 分数 p 亦良定义）。
    - "lppool1d"：x (N,C,L) → (N,C,ol)，同上 1-D，语义同 F.lp_pool1d。
    - "maxpool_idx"：x (N,C,H,W) → (out, indices) 二元组：out 语义同 F.max_pool2d
      （padding 值为 −inf，最大值并列取行优先最前者）；indices 为每窗口最大值在原输入
      （未填充）空间图上的行优先展平偏移，同 F.max_pool2d(return_indices=True)。
    - "maxunpool"：x (N,C,h,w) 小图 + indices + output_size=(H,W) → (N,C,H,W)：把每个
      窗口最大值写回 indices 指向的源位置、其余位置为 0，语义同 F.max_unpool2d。
      约束：重叠写回行为原生未定义，故要求各维 stride >= kernel（重叠配置抛 ValueError）。

    其他约定：返回新张量、不修改入参；lppool 输出与 x 同 dtype。
    """
    # TODO
    raise NotImplementedError
