import torch
import torch.nn as nn


def solve():
    """用 hook 抓中间激活、替换模块输出、改写并向上传梯度，返回完整行为账。

    固定模型（显式赋值，无随机）::

        fc1 = Linear(3,2): weight = arange(6).reshape(2,3)/10, bias = 0
        tanh = Tanh()
        fc2 = Linear(2,1): weight = [[0.3, 0.6]], bias = 0
        forward(x) = fc2(tanh(fc1(x)))
        输入 x = tensor([0.5, -1.0, 2.0], requires_grad=True)
        目标 y = tensor([1.0])
        loss = ((model(x.unsqueeze(0)).squeeze() - y)**2).sum(); loss.backward()

    固定 hook 布置::

        1. fc1 forward_hook:   计数 + 记录 out（tanh 的输入 a1）
        2. tanh forward_hook:  计数 + 记录 out（tanh 值），并 return out*2.0
           —— forward_hook 返回非 None 会替换模块输出，fc2 收到 2*tanh(a1)
        3. fc2 forward_hook:   计数 + 记录 out，另外记录 inp[0]（fc2 实际输入 h）
        4. fc1 register_full_backward_hook: 计数，return tuple(2*g + 1 for g in gin)
           —— 只改写发给**输入**的梯度（即 x.grad），参数梯度不受影响
        5. fc2 register_backward_hook（legacy）: 计数，return None（不改写）

    返回 dict::

        {"loss": float,
         "acts": {"fc1": [..2..], "tanh": [..2..], "fc2": [..1..]},  # 替换前的各层输出
         "fc2_input": [..2..],       # fc2 forward_hook 里看到的 inp[0]（= 2*tanh）
         "fired": {"fc1_fwd": .., "tanh_fwd": .., "fc2_fwd": .., "fc1_bwd": .., "fc2_bwd": ..},
         "x_grad": [..3..],          # 被 2*g+1 改写后的输入梯度
         "fc1_weight_grad": [..6..], "fc2_weight_grad": [..2..],
         "fc1_bias_grad": [..2..]}
    """
    # TODO: 装配模型与五路 hook，跑一次前反向，汇总返回
    raise NotImplementedError
