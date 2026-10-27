import torch


def solve(x, kernel_size, stride=None, dilation=1, padding=0,
          mode="unfold", output_size=None, shifts=None, dims=None):
    """unfold / fold / roll 三合一：im2col 滑窗展平、col2im 重叠累加与循环移位。

    通用参数约定：kernel_size / stride / dilation / padding 均接受 int 或
    (h, w) 二元组；stride=None 时取 kernel_size（本题约定，unfold/fold 都适用；
    注意原生 F.unfold/F.fold 的默认 stride 是 1）。H_out =
    (H + 2*ph - dh*(kh-1) - 1) // sh + 1（W_out 同理），L = H_out * W_out。
    mode 只取 "unfold" / "fold" / "roll"，其他取值抛 ValueError。

    - mode="unfold"（im2col）：x 为 (N, C, H, W)（非 4 维抛 ValueError），
      先做零填充再滑窗，返回 cols (N, C*kh*kw, L)。cols 的行按 c -> kh -> kw
      行主序排列（row = c*(kh*kw) + i*kw + j，即通道最外层、窗口内先高后宽），
      列按窗口的 hi -> wi 排列（l = hi*W_out + wi）——与 F.unfold 实测一致。
      落在输入边界外的窗口元素由零填充提供。H_out 或 W_out <= 0（kernel 大于
      pad 后尺寸）抛 RuntimeError（与原生一致）。

    - mode="fold"（col2im）：x 此时是 cols (N, C*kh*kw, L)（非 3 维抛
      ValueError），output_size=(H, W) 必填（缺省抛 ValueError），返回
      (N, C, H, W)。重叠位置求和累加；C = cols.shape[1] // (kh*kw)。
      cols.shape[1] 不被 kh*kw 整除、或 L 不恰等于按 output_size 算出的
      H_out*W_out（output_size 过大或过小都算不匹配）时抛 RuntimeError
      （与原生一致）。

    - mode="roll"：循环移位，逐维 (i + shift) mod n 语义（取模，支持负
      shifts），kernel_size / stride / dilation / padding / output_size 全部
      忽略。shifts 必填（None 抛 ValueError）：
      * shifts 为 int、dims 为 None：对最后一维 roll shifts 步（本题约定；
        注意原生 torch.roll 在 dims=None 时是先展平再 roll，本题不展平）。
      * shifts 为 int、dims 为 int：对该维 roll。
      * shifts 为 int、dims 为 (d1, d2, ...) 元组：只对 dims[0] 这一维 roll
        （本题约定；原生 torch.roll 不接受这种组合）。
      * shifts 为 (s1, s2, ...) 元组：dims 必须是等长元组，逐维分别 roll
        shifts[i] 步；dims 为 None 或长度不匹配抛 ValueError。
      * dims 越界抛 IndexError。

    三种模式都返回新张量，不修改输入；除上述注明 ValueError / IndexError /
    RuntimeError 的情形外，不做额外校验。
    """
    # TODO
    raise NotImplementedError
