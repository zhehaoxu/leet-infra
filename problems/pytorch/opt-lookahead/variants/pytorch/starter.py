import torch


def solve(params, base="sgd", lr=0.05, momentum=0.9, k=2, alpha=0.5, betas=(0.9, 0.999), eps=1e-8):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 **Lookahead 封装器** 实例（Zhang et al., arXiv:1710.08983），
    #   它在内部持有并驱动一个真正的 base 优化器，step() 就地更新 params 里的张量本身。
    #
    # 入参约定（全部按关键字传入，名字与默认值须一致）：
    #   params    : 待优化张量列表（同一块内存，就地改；也是 self.param_groups[0]["params"]）
    #   base      : "sgd" → torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0,
    #                        dampening=0, nesterov=False)
    #               "adam"→ torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=0,
    #                        amsgrad=False)
    #               其余取值抛 ValueError
    #   k         : 正整数，每走 k 个 fast 步同步一次
    #   alpha     : ∈ [0,1] 的浮点，慢权重吸收比例
    #
    # 必须暴露的属性（评测要读）：
    #   .base_optimizer : 那个内部 base 优化器对象（评测用它的 .state[p] 探针查动量/二阶矩）
    #   .slow_weights   : 与 params 一一对应的列表，每项是与对应参数同形状同 dtype 的张量
    #
    # step() 的三步顺序（下标约定：构造时 t=0、s_i = params[i] 的 clone；t 只数 step() 调用次数）：
    #   1) 先让 base 走一步：base_optimizer.step()   # fast 权重 θ 就地更新
    #      —— base 自身的状态（SGD 的 momentum_buffer、Adam 的 exp_avg/exp_avg_sq/step）
    #         照常累积，同步**绝不**清零、重置或重建这些状态，也不重新实例化 base。
    #   2) 再推进计数：t ← t + 1
    #   3) 仅当 t 是 k 的正整数倍时同步，且两行顺序不可颠倒：
    #        s ← s + alpha * (θ − s)     # 慢更新在前
    #        θ ← s                       # 重置在后，就地写回训练权重（copy_，不是新建张量）
    #      即第一次同步发生在第 k 次 step() **之后**（不是之前、也不是第 k+1 次的开头）；
    #      同步由 t 对 k 取模驱动，与某个参数这一步有没有梯度无关（grad 为 None 的参数
    #      不会被 base 更新，但到点仍会被拉回自己的慢权重）。
    #
    # 三个退化口径（自检用）：alpha=1 → 轨迹与裸 base 完全一致；alpha=0 → 每次同步把 θ 硬回滚到
    #   上一次同步点（慢权重从不吸收新东西）；k=1 → 每步同步，等价于把 base 每步增量整体缩放 alpha。
    # 全程 float32、禁止引入随机性；不许用现成的 Lookahead 实现。
    raise NotImplementedError
