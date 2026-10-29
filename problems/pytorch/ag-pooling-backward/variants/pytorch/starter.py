import torch


def solve(x, dY, mode='max', kernel=2, stride=None, padding=0):
    """2D 池化的反向：只返回 dx（池化没有可学参数）。

    入参：
    - x: (B, C, H, W) 前向输入；dY: (B, C, OH, OW) 上游梯度。
    - mode: 'max' | 'avg' | 'adaptive_avg'，其它值抛 ValueError。
    - kernel: int 或 (kh, kw)；stride=None 时等于 kernel；padding 只允许用于 'max'。
    - 'adaptive_avg' 忽略 kernel/stride/padding，输出尺寸由 dY 的空间形状决定。

    口径：
    - 'max'：把 dY[b,c,i,j] 整份交给窗口内取到最大值的**那一个**位置；并列最大值时
      交给行主序最小（先上后左）的位置——torch 的并列行为未定义，本题以此口径为准。
    - 'avg'：窗口内每个位置分得 dY / (kh·kw)；要求 padding=0，否则抛 ValueError。
    - 'adaptive_avg'：第 i 个窗口覆盖 [floor(i·H/OH), ceil((i+1)·H/OH))，窗高 = 该区间长度，
      每个位置分得 dY / (窗高·窗宽)。窗口可能重叠且长度不等，不能一律除以 kernel 面积。
    - 窗口完全落在 x 外的部分（padding 区）不产生 dx；dx 形状恒等于 x。
    - 输入为 int 也无所谓：内部 float64 计算，返回 dtype 与 x 一致。
    """
    # TODO
    raise NotImplementedError
