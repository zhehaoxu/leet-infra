import torch
import torch.nn as nn


def solve():
    """全冻结 -> 换 head -> 优化器只收可训参数 -> 训练 3 步，返回全部行为账目。

    固定构造（数值均不依赖随机数）::

        _explicit(shape, start) = (arange(n) + start).reshape(shape) / 100   # n = numel(shape)

        class Net(nn.Module):                       # forward: head(encoder(x))
            encoder = nn.Sequential(nn.Linear(8,6), nn.Tanh(), nn.Linear(6,6))
            head    = nn.Linear(6, 4)
            # 显式赋值：encoder[0].weight=_explicit((6,8),0), bias 全 0
            #          encoder[2].weight=_explicit((6,6),48), bias 全 0
            #          head.weight=_explicit((4,6),84),       bias 全 0

    流程::

        1. total_before = 全部参数 numel 之和（冻结前统计）
        2. 对所有参数 requires_grad_(False)；trainable_before = 可训 numel 之和
        3. 换 head：new = nn.Linear(6,5)，weight=_explicit((5,6),0)、bias 全 0 显式赋值
           后替换 model.head（新 head 默认 requires_grad=True）
        4. total_after / trainable_after 同上统计
        5. flags = [[参数名, requires_grad], ...] 按 named_parameters() 顺序
        6. opt = SGD(只收集 requires_grad 的参数, lr=0.1, momentum=0.9)
        7. x = arange(16).reshape(2,8)/8；y = arange(10).reshape(2,5)/10
        8. 循环 3 步：zero_grad -> pred=model(x) -> loss=((pred-y)**2).mean()
           -> 记录 loss.item() -> backward -> step
        9. 步后检查：冻结参数与步前 clone 逐位相同 (torch.equal)；
           冻结参数 .grad 是否仍为 None；len(opt.state) 为优化器建了状态的参数个数

    返回 dict::

        {
          "total_before": int, "total_after": int,
          "trainable_before": int, "trainable_after": int,
          "flags": list[[str, bool]],
          "losses": list[float],                       # 3 项，backward 前记录
          "final_head_weight_flat": list[float],       # 5x6 行主序展平
          "final_head_bias_flat": list[float],
          "frozen_bitwise_identical": bool,
          "frozen_grad_is_none": bool,
          "n_opt_state_params": int,
        }
    """
    # TODO: 按 docstring 流程装配账目
    raise NotImplementedError
