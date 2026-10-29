import torch


def solve(mask, counts, size):
    """COCO RLE 编解码 + 不物化掩码的 mask IoU。

    入参：
    - mask: (H, W) 的 bool 或 uint8 张量，元素只能是 0/1，表示掩码 A。
    - counts: 一维整型张量（int64 或 int32），掩码 B 的 COCO RLE 游程编码。
      约定：按**列优先**（Fortran）顺序扫描 (H, W) 的 H*W 个位置，游程值从 0
      （背景）开始交替；counts[0] 是第一个前景像素之前的背景长度（可以为 0），
      counts[1] 是紧接着的一段前景长度，依此类推，最后一段可以是背景也可以是
      前景。要求：元素非空、全部 >= 0、首项可以为 0 而**其余各项必须 > 0**
      （规范形式，不允许内部零长游程）、所有元素之和等于 H*W。
    - size: 长度 2 的整数序列 (H, W)，RLE 的尺寸声明；必须与 mask.shape 完全
      一致，且 H、W 均 >= 1（判题据此核对 counts 的尺寸归属，不做隐式广播）。

    返回四元组：
    - iou: 0 维 float64 张量，= intersection / union；union 为 0 时取 0.0。
    - intersection: 0 维 int64 张量，A、B 同时为前景的像素数。
    - union: 0 维 int64 张量，A 或 B 为前景的像素数（= |A| + |B| - 交集）。
    - counts_a: 一维 int64 张量，mask 的**规范** RLE 编码（与入参 counts 同一
      约定）。全背景编码为 [H*W]；列优先扫描的第一个像素就是前景时首项写 0；
      末尾若为背景，背景游程必须包含在内。

    口径要点：
    - 列优先：位置线性下标 lin = x * H + y（x 为列、y 为行），即先走完整整一列
      再进入下一列。行优先编码在同一 mask 上会得到不同的 counts。
    - IoU 只依赖两份游程；实现路径不限，但判题规模下把 counts 展开成稠密掩码
      再比较也能通过，是否归并只影响复杂度，不影响结果。
    - 不修改任何输入张量。

    非法输入一律抛 ValueError：mask 不是 torch.Tensor 或维度不是 2、dtype 不是
    bool/uint8、含 0/1 以外的值；counts 不是一维整型张量、为空、含负值、除首项
    外出现 0、元素和不等于 H*W；size 不是长度 2 的正整数序列或与 mask.shape
    不一致。
    """
    # TODO
    raise NotImplementedError
