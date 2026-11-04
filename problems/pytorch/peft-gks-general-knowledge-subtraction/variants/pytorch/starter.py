"""GenKnowSub: general-knowledge subtraction for Arrow libraries (official
PEFT arrow.py gen_know_sub + create_arrow_model gate; paper arXiv:2505.10939,
ACL 2025, official saharsamr/Modular-LLM).

GenKnowSub disentangles task-specific and general knowledge: the A/B factors
of the gks_ (general-domain) adapters are AVERAGED, and the averages are
subtracted IN PLACE from each task adapter's factors ("forgetting-via-negation"
task arithmetic). The purified library is then routed by Arrow. The gate is a
no-op when use_gks is False or when everything is already purified and nothing
was added; the first run purifies ALL task adapters, later runs only the newly
added ones. Because the subtraction acts on the FACTORS, the task delta
(B-avgB)@(A-avgA) carries cross terms — it is NOT B@A - avgB@avgA.

Accounts (official, verbatim):
- gks_average: avg_A = stack(A_gks).mean(0), avg_B = stack(B_gks).mean(0)
- gks_subtract: lora_A[name].weight.data.sub_(avg_A) (pure-function clone here),
  limited to gks_added_adapter_names on incremental runs
- gks_gate: 'skip' / 'all' / 'added'
- require_gks_paths: use_gks with no general paths -> ValueError
- task_delta: einsum("eor,erf->eof", b_sub, a_sub)
"""
import torch


def gks_average(g_a, g_b):
    raise NotImplementedError


def gks_subtract(a_stack, b_stack, avg_a, avg_b, only=None):
    raise NotImplementedError


def gks_gate(use_gks, gks_done, n_added):
    raise NotImplementedError


def require_gks_paths(use_gks, n_paths):
    raise NotImplementedError


def task_delta(a_sub, b_sub):
    raise NotImplementedError


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
