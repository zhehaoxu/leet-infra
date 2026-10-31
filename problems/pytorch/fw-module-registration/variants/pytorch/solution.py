import torch
import torch.nn as nn


def solve(in_dim, hidden, out_dim):
    """按 starter 注释的 9 步搭模块，逐步记录注册机制账本。"""
    for name, v in (("in_dim", in_dim), ("hidden", hidden), ("out_dim", out_dim)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name} must be a positive integer")

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden)
            shared = nn.Parameter(torch.zeros(out_dim))
            self.p_first = shared
            self.p_second = shared
            self.register_buffer("running", torch.zeros(hidden), persistent=True)
            self.register_buffer("scratch", torch.ones(2), persistent=False)
            self.register_parameter("placeholder", None)
            self.loose = [nn.Linear(hidden, 2)]
            self.blocks = nn.ModuleList([nn.Linear(hidden, out_dim)])

    m = Mixed()

    ledger = {}

    named = list(m.named_parameters())
    ledger["param_names_dedup"] = [n for n, _ in named]
    ledger["param_count_dedup"] = len(named)
    named_nd = list(m.named_parameters(remove_duplicate=False))
    ledger["param_names_nodup"] = [n for n, _ in named_nd]
    ledger["param_count_nodup"] = len(named_nd)
    ledger["shared_same_object"] = m.p_first is m.p_second
    ledger["dedup_keeps_first"] = "p_first" in ledger["param_names_dedup"] and (
        "p_second" not in ledger["param_names_dedup"]
    )

    ledger["buffer_names"] = [n for n, _ in m.named_buffers()]

    sd = m.state_dict()
    keys = list(sd.keys())
    ledger["state_dict_keys"] = keys
    ledger["sd_has_duplicate_key"] = "p_first" in keys and "p_second" in keys
    ledger["nonpersistent_in_sd"] = "scratch" in keys
    ledger["placeholder_in_named"] = "placeholder" in ledger["param_names_nodup"]
    ledger["placeholder_in_sd"] = "placeholder" in keys

    mods = list(m.modules())
    ledger["loose_in_modules"] = any(
        mod is m.loose[0] for mod in mods
    )
    ledger["module_count"] = len(mods)
    ledger["module_linear_count"] = sum(1 for mod in mods if isinstance(mod, nn.Linear))

    # 实验 1：持久 buffer 属性重新赋普通 Tensor，注册仍在
    m.running = torch.randn(hidden)
    ledger["buffer_reassign_kept"] = ("running" in dict(m.named_buffers())) and (
        "running" in m.state_dict()
    )

    # 实验 2：buffer 跟随 .to()
    m.to(torch.float64)
    ledger["buffer_follows_to"] = m.running.dtype == torch.float64

    return ledger
