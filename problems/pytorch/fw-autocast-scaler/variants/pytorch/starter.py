import torch


def solve():
    """CPU 上 fp16 autocast + GradScaler 训练循环：记全 dtype、scale 升降、跳步与权重账。

    固定构造（数值全部二进制可精确表示，fp16 无舍入损失，可用纯 float 重算）::

        x = tensor([[1.5, -0.5]]);  w = tensor([0.5, 0.25], requires_grad=True)
        y = tensor(1.0);            opt = SGD([w], lr=0.25)
        scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=16.0,
                                      growth_factor=2.0, backoff_factor=0.5, growth_interval=2)

    循环 t = 0..5，每步顺序::

        1. opt.zero_grad()
        2. with torch.amp.autocast("cpu", dtype=torch.float16):
               out = x @ w
               loss = ((out - y) ** 2).mean()
        3. t==0：记录 str(out.dtype) / str(loss.dtype)
        4. scaler.scale(loss).backward()；t==0：记录 str(w.grad.dtype)，并做
           「次序违规探针」：先 scaler.unscale_(opt)，再 try scaler.unscale_(opt)
           第二次必须抛 RuntimeError（catch 记 True，之后照常继续）
        5. t in (1, 4)：注入溢出 w.grad[0] = float("inf")
        6. before = w.detach().clone()；scaler.step(opt)；
           updated[t] = not torch.equal(before, w.detach())
        7. scaler.update()；scale_traj[t] = float(state_dict()["scale"])，
           trackers[t] = int(state_dict()["_growth_tracker"])
        8. losses[t] = float(loss.item())

    返回 dict::

        out_dtype / loss_dtype / grad_dtype: str      # torch.float16 / torch.float16 / torch.float32
        scale_traj: list[float]                        # 每步 update 后的 scale
        trackers:   list[int]                          # 每步 update 后的 _growth_tracker
        updated:    list[bool]                         # 该步权重是否真的被更新
        losses:     list[float]
        n_skipped:  int                                # 6 - sum(updated)
        final_w_flat: list[float]
        double_unscale_raises: bool

    GradScaler 状态机（可独立推导）：发现 inf/nan → 跳过 step、
    \(scale\leftarrow scale\times0.5\)、tracker 清零；否则 \(tracker{+}{=}1\)，
    当 \(tracker==growth\_interval\) 时 \(scale\leftarrow scale\times2\) 且清零。
    """
    # TODO: 按 docstring 跑循环并汇总账目
    raise NotImplementedError
