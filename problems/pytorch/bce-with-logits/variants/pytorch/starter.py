import torch


def solve(logits, targets, weight=None, reduction="mean", pos_weight=None):
    """bce-with-logits：F.binary_cross_entropy_with_logits 等价实现，输入是 logits 而非概率。

    入参：
    - logits: float32 张量 x（未过 sigmoid）。
    - targets: 与 logits 同形的 0/1 标签（float）。
    - weight: None 或与逐元素损失可广播的 float 张量，逐元素乘在损失上。
    - reduction: "mean"（默认）/ "sum" / "none"。
      * "none" 返回逐元素损失（已乘 weight），形状与 logits 相同；
      * "sum" 返回全部元素之和（0 维）；
      * "mean" 返回 全部元素之和 ÷ logits.numel()——分母是元素个数，
        **不是** Σweight（这是与手写「加权平均」最容易吵起来的一条口径，务必照此实现）。
    - pos_weight: None 或可与 logits 广播的张量，给正类项加权重 w：
      loss_i = pos_weight_i·y_i·log(1+e^{−x_i}) + (1−y_i)·log(1+e^{x_i})。

    数值要求：禁止先 sigmoid 再 log（x=−1e3 时 sigmoid 下溢为 0，log 0 = −inf）。
    必须用稳定式，逐元素写成（记 L = log(1 + e^{−|x|})，w 缺省为 1）
      loss_i = (1−y_i)·max(x_i, 0) + w_i·y_i·(max(x_i, 0) − x_i)
               + (1 + y_i·(w_i − 1))·L
    w=1 时即标准形式 max(x,0) − x·y + L。内部用 float64 计算，返回 float32。
    reduction 取值非法时抛 ValueError。
    """
    # TODO
    raise NotImplementedError
