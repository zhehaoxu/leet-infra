import torch
import torch.nn as nn


def solve():
    """断点续训连续性账：model+optimizer+scheduler+RNG 全量存档恢复的逐位验证（CPU-only）。

    固定构造（参数用显式张量创建，构造过程不消耗全局 RNG)::

        class Net(nn.Module):            # forward: tanh 后接第二个线性层
            fc1_w = nn.Parameter([[0.5, -0.25], [0.125, 0.25]])   # 形状 (2, 2)
            fc1_b = nn.Parameter([0.0, 0.5])
            fc2_w = nn.Parameter([[0.25, -0.5]])                  # 形状 (1, 2)
            fc2_b = nn.Parameter([0.0])
            forward(x) = F.linear(tanh(F.linear(x, fc1_w, fc1_b)), fc2_w, fc2_b)

    每条路径先 ``torch.manual_seed(1234)``，再新建一套对象::

        model = Net()
        opt   = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    单步训练循环顺序（必须严格照此)::

        opt.zero_grad()
        x = torch.randn(1, 2); y = torch.randn(1)
        loss = ((model(x) - y) ** 2).sum()      # 记录 loss.item() 到当前列表
        loss.backward()
        opt.step()
        sched.step()

    三条路径::

        路径 1（黄金轨迹）：连续跑 6 步 -> losses_full。
        路径 2（完整恢复）：跑 3 步后在内存里存档
            ckpt = {"model": {k: v.clone() for k, v in model.state_dict().items()},
                    "opt":   <优化器 state_dict 的逐张量深拷贝>,
                    "sched": sched.state_dict(),
                    "rng":   torch.get_rng_state().clone(),
                    "step":  3}
            注意：Optimizer.state_dict() 返回的张量与活体 momentum_buffer 是同一对象，
            且 load_state_dict 会把传入张量直接绑进新优化器（CPU 上不拷贝）；
            存档与每次加载都必须对 state 里的张量做新克隆，否则两次恢复共享同一
            buffer、后一条路径的原地更新会污染 ckpt 与前一条路径的账。
            随后 torch.randn(2) 消耗 2 个无关随机数（模拟崩溃前的其他随机行为），
            再新建全新 model/opt/sched，加载 ckpt 的 model/opt/sched 并
            torch.set_rng_state(ckpt["rng"])，从第 3 步续跑到第 6 步。
            losses_resumed = 前 3 步记录 + 续跑 3 步记录（共 6 项）。
        路径 3（漏恢复 RNG）：与路径 2 结构完全相同、同样 torch.randn(2) 垃圾消耗、
            同样加载 ckpt，唯独不调用 set_rng_state -> losses_no_rng（6 项）。

    返回 dict::

        {
          "losses_full": list[float],          # 路径 1 的 6 个 loss.item()
          "losses_resumed": list[float],       # 路径 2 的 6 个 loss.item()
          "losses_no_rng": list[float],        # 路径 3 的 6 个 loss.item()
          "resumed_bitwise_identical": bool,   # losses_resumed == losses_full（Python 列表相等）
          "no_rng_tail_diverges": bool,        # losses_no_rng[3:] != losses_full[3:]
          "final_weight_identical": bool,      # 路径 2 结束时全部参数张量与路径 1 结束时
                                               #   逐张量 torch.equal（同形状，不展平）
          "momentum_restored": bool,           # 路径 2 结束时 fc1_w 的 momentum_buffer
                                               #   与路径 1 结束时 torch.equal
          "sched_last_epoch_resumed": int,     # 路径 2 结束时 sched.last_epoch
        }
    """
    # TODO: 按 docstring 跑三条路径并装配连续性账
    raise NotImplementedError
