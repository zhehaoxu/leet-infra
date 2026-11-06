"""Reference: L4 design review over the frozen stage vocabulary (step 06)."""


def solve(decisions, trace):
    from scaffold.dialog_runner import DATA_FLOW_STAGES, audit_stage_order

    order_ok, reason = audit_stage_order(decisions)
    stages = list(DATA_FLOW_STAGES)
    if not order_ok:
        accepted, reason = False, reason
    elif list(trace or []) != stages:
        accepted, reason = False, "trace"
    else:
        accepted, reason = True, "ok"
    return {
        "accepted": accepted,
        "reason": reason,
        "stages": stages,
        "trace_len": len(trace or []),
    }
