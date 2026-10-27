import torch


def solve(x, r, mode="shuffle"):
    """通道↔空间双射重排四合一：pixel_shuffle / pixel_unshuffle / space-to-depth / depth-to-space。

    入参：
    - x: float 张量（CPU，float32 / float64），4 维 (N, C, H, W)。允许非连续输入
      （如 transpose 得到的视图），结果必须与同值连续输入完全一致。
    - r: 缩放因子，int 类型且 r >= 1；非 int（含 float / bool）或 r < 1 抛 ValueError。
    - mode: str，以下四选一，其余取值抛 ValueError：
      * "shuffle"：x (N, C·r², H, W) → (N, C, H·r, W·r)。语义同 F.pixel_shuffle：
        out[n, c, h*r + i, w*r + j] = x[n, c*r² + i*r + j, h, w]（0 <= i, j < r）。
        要求通道数能被 r² 整除，否则抛 ValueError（空间维只放大，无整除要求）。
      * "unshuffle"：shuffle 的逆，x (N, C, H, W) → (N, C·r², H/r, W/r)，语义同
        F.pixel_unshuffle：out[n, c*r² + i*r + j, h, w] = x[n, c, h*r + i, w*r + j]。
        要求 H、W 均能被 r 整除，否则抛 ValueError。
      * "s2d"：space-to-depth，x (N, C, H, W) → (N, C·r², H/r, W/r)。在 channels-first
        布局下与 F.pixel_unshuffle 完全等价（输出通道序同为 c*r² + i*r + j，可对拍验证），
        整除要求同 "unshuffle"。
      * "d2s"：s2d 的逆，x (N, C·r², H, W) → (N, C, H·r, W·r)，与 F.pixel_shuffle 等价，
        通道数整除约束同 "shuffle"。

    实现约束（本题考点）：不得调用 F.pixel_shuffle / F.pixel_unshuffle，必须用
    view/reshape/permute 或显式索引重排完成该双射。

    其他约定：
    - x 非 4 维抛 ValueError。
    - r == 1 时四种 mode 均为严格恒等（torch.equal(y, x)）。
    - 往返恒等：unshuffle(shuffle(x)) == x、d2s(s2d(x)) == x（torch.equal）。
    - 返回张量与 x 同 dtype；不修改入参。
    """
    # TODO
    raise NotImplementedError
