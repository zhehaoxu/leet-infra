"""参考答案：model+optimizer+scheduler+RNG 全量内存存取，第 3 步恢复与一口气跑 6 步逐位一致。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

JUNK_DRAWS = 2  # 存档后混入的无关随机消耗，用于检验 RNG 账是否被恢复


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_w = nn.Parameter(torch.tensor([[0.5, -0.25], [0.125, 0.25]]))
        self.fc1_b = nn.Parameter(torch.tensor([0.0, 0.5]))
        self.fc2_w = nn.Parameter(torch.tensor([[0.25, -0.5]]))
        self.fc2_b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        a = torch.tanh(F.linear(x, self.fc1_w, self.fc1_b))
        return F.linear(a, self.fc2_w, self.fc2_b)


def make_trio():
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    return model, opt, sched


def run_steps(model, opt, sched, start, end, losses):
    for i in range(start, end):
        opt.zero_grad()
        x = torch.randn(1, 2)
        y = torch.randn(1)
        pred = model(x)
        loss = ((pred - y) ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()
        sched.step()


def _clone_opt_sd(opt_sd):
    """深拷贝优化器 state_dict：Optimizer.state_dict() 返回的张量与活体
    buffer 是同一对象，load_state_dict 又会把这些对象直接绑进新优化器，
    两次恢复共享一份 ckpt 会互相原地污染，必须逐张量 clone。"""
    return {
        "state": {
            i: {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in st.items()}
            for i, st in opt_sd["state"].items()
        },
        "param_groups": [{**g, "params": list(g["params"])}
                         for g in opt_sd["param_groups"]],
    }


def solve():
    # —— 路径 1：一口气 6 步（黄金轨迹）
    torch.manual_seed(1234)
    m, o, s = make_trio()
    losses_full: list[float] = []
    run_steps(m, o, s, 0, 6, losses_full)
    final_w_full = [p.detach().clone() for p in m.parameters()]
    mom_full = o.state_dict()["state"][0]["momentum_buffer"].clone()

    # —— 路径 2：3 步存档 -> 全新对象恢复（含 RNG）-> 再跑 3 步
    torch.manual_seed(1234)
    m, o, s = make_trio()
    losses_seg1: list[float] = []
    run_steps(m, o, s, 0, 3, losses_seg1)
    ckpt = {
        "model": {k: v.clone() for k, v in m.state_dict().items()},
        "opt": _clone_opt_sd(o.state_dict()),
        "sched": s.state_dict(),
        "rng": torch.get_rng_state().clone(),
        "step": 3,
    }
    torch.randn(JUNK_DRAWS)  # 崩溃前同进程里的无关随机消耗
    m2, o2, s2 = make_trio()  # 全新对象，随后被 checkpoint 覆盖
    m2.load_state_dict(ckpt["model"])
    o2.load_state_dict(_clone_opt_sd(ckpt["opt"]))
    s2.load_state_dict(ckpt["sched"])
    torch.set_rng_state(ckpt["rng"])
    losses_seg2: list[float] = []
    run_steps(m2, o2, s2, ckpt["step"], 6, losses_seg2)
    losses_resumed = losses_seg1 + losses_seg2
    final_w_resumed = [p.detach().clone() for p in m2.parameters()]
    mom_resumed = o2.state_dict()["state"][0]["momentum_buffer"].clone()
    sched_last_resumed = s2.last_epoch

    # —— 路径 3：同样结构、同样垃圾消耗，唯独**不**恢复 RNG
    torch.manual_seed(1234)
    m, o, s = make_trio()
    losses_seg1b: list[float] = []
    run_steps(m, o, s, 0, 3, losses_seg1b)
    torch.randn(JUNK_DRAWS)
    m3, o3, s3 = make_trio()
    m3.load_state_dict(ckpt["model"])
    o3.load_state_dict(_clone_opt_sd(ckpt["opt"]))
    s3.load_state_dict(ckpt["sched"])
    # 不 set_rng_state：数据流错位
    losses_seg2b: list[float] = []
    run_steps(m3, o3, s3, 3, 6, losses_seg2b)
    losses_no_rng = losses_seg1b + losses_seg2b

    return {
        "losses_full": losses_full,
        "losses_resumed": losses_resumed,
        "losses_no_rng": losses_no_rng,
        "resumed_bitwise_identical": losses_resumed == losses_full,
        "no_rng_tail_diverges": losses_no_rng[3:] != losses_full[3:],
        "final_weight_identical": all(
            torch.equal(a, b) for a, b in zip(final_w_full, final_w_resumed)),
        "momentum_restored": bool(torch.equal(mom_full, mom_resumed)),
        "sched_last_epoch_resumed": sched_last_resumed,
    }
