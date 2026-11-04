"""PEFT: GenKnowSub general-knowledge subtraction (official PEFT arrow.py
gen_know_sub + create_arrow_model gate; paper arXiv:2505.10939, ACL 2025,
official saharsamr/Modular-LLM).

Official account:
  1. gen_know_sub averages the A/B factors over the gks_ adapters and
     subtracts the averages IN PLACE from each task adapter's factors:
     avg_A = stack(A_gks).mean(0), avg_B = stack(B_gks).mean(0);
     lora_A[name].weight.data.sub_(avg_A), same for B. This purifies the task
     adapters via "forgetting-via-negation" (task arithmetic, 2212.04089).
  2. Gate: a no-op when use_gks is False, or when gks_done and no adapters
     were added; "all" task adapters are purified on the first run
     (gks_done=False), only the newly added ones afterwards.
  3. create_arrow_model: use_gks=True with no general paths raises ValueError.
  4. The subtraction acts on the FACTORS, so the resulting task delta
     (B-avg_B)@(A-avg_A) is NOT B@A - avg_B@avg_A — it carries cross terms.
"""
import torch


def gks_average(g_a, g_b):
    """Official step 1: mean of the stacked general-knowledge factors."""
    avg_a = torch.stack(list(g_a), dim=0).mean(0)
    avg_b = torch.stack(list(g_b), dim=0).mean(0)
    return avg_a, avg_b


def gks_subtract(a_stack, b_stack, avg_a, avg_b, only=None):
    """Official step 2: pure-function equivalent of the in-place
    weight.data.sub_ — clone then subtract; only limits the purified subset
    (gks_added_adapter_names)."""
    a_new = a_stack.clone()
    b_new = b_stack.clone()
    if only is None:
        a_new = a_new.sub_(avg_a)
        b_new = b_new.sub_(avg_b)
    else:
        idx = torch.as_tensor(sorted(only), dtype=torch.long)
        a_new[idx] = a_new[idx].sub_(avg_a)
        b_new[idx] = b_new[idx].sub_(avg_b)
    return a_new, b_new


def gks_gate(use_gks, gks_done, n_added):
    """Official gate of gen_know_sub: 'skip' (no-op), 'all' (first run), or
    'added' (only the newly added adapters)."""
    if not use_gks or (gks_done and n_added == 0):
        return "skip"
    if gks_done:
        return "added"
    return "all"


def require_gks_paths(use_gks, n_paths):
    """Official create_arrow_model check: general paths required for gks."""
    if use_gks and n_paths == 0:
        raise ValueError("You should provide general LoRA paths if you want to use GenKnowSub.")
    return n_paths


def task_delta(a_sub, b_sub):
    """Per-adapter task delta after subtraction: (B-avgB)@(A-avgA) per expert."""
    return torch.einsum("e o r, e r f -> e o f", b_sub, a_sub)


def solve(mode=None, **kw):
    if mode == "gks_average":
        return gks_average(kw["g_a"], kw["g_b"])
    if mode == "gks_subtract":
        return gks_subtract(kw["a_stack"], kw["b_stack"], kw["avg_a"], kw["avg_b"],
                            only=kw.get("only"))
    if mode == "gks_gate":
        return gks_gate(kw["use_gks"], kw["gks_done"], kw["n_added"])
    if mode == "require_gks_paths":
        return require_gks_paths(kw["use_gks"], kw["n_paths"])
    if mode == "task_delta":
        return task_delta(kw["a_sub"], kw["b_sub"])
    raise ValueError(f"unknown mode {mode!r}")
