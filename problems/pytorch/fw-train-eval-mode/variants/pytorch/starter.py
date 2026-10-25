import torch
import torch.nn as nn


def solve():
    """train/eval 模式、推理与 clip 的行为账（CPU-only）。按阶段返回单个 dict。

    阶段 A —— BatchNorm1d running 统计账::

        bn = nn.BatchNorm1d(3)；bn.train()（默认 affine=True、momentum=0.1、eps=1e-5，
        running_mean=0、running_var=1、num_batches_tracked=0）
        x1 = torch.arange(6).reshape(2, 3) / 4.0
        x2 = torch.tensor([[-1.0, 2.0, 0.5], [1.5, -0.5, 3.0]])
        依次 bn(x1)、bn(x2)。train 前向更新规则（对每个通道 c）：
            running_mean <- (1-0.1)*running_mean + 0.1*batch_mean
            running_var  <- (1-0.1)*running_var  + 0.1*batch_var   # batch_var 为无偏(除 N-1)
            num_batches_tracked += 1（仅 train 模式）

    阶段 B —— eval 前向不动任何状态::

        记录 bn.state_dict() 各张量的 clone；bn.eval()；
        xe = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]])；out_eval = bn(xe)。
        eval 输出用 running 统计：out = weight * (x - running_mean)/sqrt(running_var + 1e-5) + bias。
        eval_state_unchanged = 前向后 state_dict 与记录 clone 全部 torch.equal；
        eval_out_requires_grad = out_eval.requires_grad（输入无梯度，但仿射参数 requires_grad=True）。

    阶段 C —— Dropout 开关账::

        torch.manual_seed(7)；drop = nn.Dropout(0.25)；xd = torch.ones(4, 5)
        drop.train() 后 yd = drop(xd)：值域恰为 {0, 1/0.75}（倒置缩放），
            dropout_train_unique = 去重升序 list[float]；n_dropout_zeros = 0 的个数
        drop.eval() 后 drop(xd) 与 xd torch.equal（恒等）。

    阶段 D —— eval() 不等于 no_grad；inference_mode 张量的性质::

        xg = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.inference_mode(): yi = xg * 2.0
        inference_requires_grad = yi.requires_grad                    # False
        inference_backward_raises = yi.sum().backward() 是否抛 RuntimeError
        inference_enable_raises = 模式外 yi.requires_grad_(True) 是否抛 RuntimeError
        with torch.no_grad(): zn = xg * 2.0
        no_grad_enable_ok = zn.requires_grad_(True) 成功且 zn.requires_grad == True

    阶段 E —— clip_grad_norm_(norm_type=inf) 均匀缩放账::

        w1 = nn.Parameter([[3., -4.], [0., 2.]])；w2 = nn.Parameter([1., -0.5])
        手工置 w1.grad / w2.grad 为各自 clone；
        total = torch.nn.utils.clip_grad_norm_([w1, w2], max_norm=1.0, norm_type=inf)
        norm_type=inf 时 total = 全部梯度的最大绝对值；clip 系数
        coef = max_norm / (total + 1e-6)，对每一个梯度元素统一相乘。

    返回 dict::

        {
          "running_mean_after2": list[float],     # 通道 3 项，阶段 A 后
          "running_var_after2": list[float],
          "num_batches_train": int,               # 阶段 A 后应为 2
          "eval_state_unchanged": bool,
          "eval_out_requires_grad": bool,
          "eval_out_flat": list[float],           # out_eval 行主序展平，6 项
          "dropout_train_unique": list[float],    # [0.0, 1/0.75 的 float32 值]
          "n_dropout_zeros": int,
          "dropout_eval_identity": bool,
          "inference_requires_grad": bool,
          "inference_backward_raises": bool,
          "inference_enable_raises": bool,
          "no_grad_enable_ok": bool,
          "total_norm_inf": float,                # clip 前 total
          "grads_after_flat": list[float],        # w1 展平 4 项 + w2 的 2 项
          "clip_scaled_uniformly": bool,          # 逐元素与 pre*coef 全部 allclose(1e-5,1e-5)
        }
    """
    # TODO: 按 docstring 五阶段装配行为账
    raise NotImplementedError
