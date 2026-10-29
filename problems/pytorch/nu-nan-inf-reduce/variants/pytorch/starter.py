import torch


def solve(x, dim=None, mode='mean', keepdim=False):
    """沿 dim 做「跳过 NaN」的归约，但 inf 是合法数值要照常参与。不许调用 torch.nan* 系列。

    入参：
    - x: float32 或 float64 张量，输出保持同样的 dtype。
    - dim: None 表示整体归约（返回 0-D 张量）；否则为 int，允许负轴，需先归一化。
      越界（归一化后不在 [0, ndim) 内）抛 ValueError。
    - mode: 'sum' | 'mean' | 'max' | 'min'，其它值抛 ValueError。
    - keepdim: 只接受 bool，True 时保留被归约的轴（长度 1）。

    语义（把每个待归约段记为 seg，valid = seg 中不是 NaN 的元素）：
    - 'sum'  = Σ valid，**全 NaN 段给 0.0**（NaN 当加数被丢掉，不污染）。
    - 'mean' = Σ valid / len(valid)，分母是 valid 的个数而不是段的长度；
               **全 NaN 段给 NaN**。
    - 'max'  = max(valid)，全 NaN 段给 NaN。
    - 'min'  = min(valid)，全 NaN 段给 NaN。
    - +inf / -inf 不是 NaN：它们参与计算，所以含 +inf 的段 max 是 +inf，
      同时含 +inf 和 -inf 的段 min 是 -inf。
    - 段里既有 NaN 又有 inf 时，NaN 被跳过，inf 决定结果。
    - 被归约轴长度为 0（空张量）同样看做 count == 0：'sum' 给 0.0，其余给 NaN。

    输出形状：dim=None 时为 0-D；否则删掉 dim（keepdim=True 时该轴长度记为 1）。
    非浮点输入抛 ValueError。
    """
    # TODO
    raise NotImplementedError
