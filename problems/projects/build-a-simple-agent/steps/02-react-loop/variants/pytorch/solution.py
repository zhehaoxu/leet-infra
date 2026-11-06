"""Reference: ReAct loop state machine (step 02).

Uses the frozen step-01 module; model replies are fixture arguments.
"""


def solve(script, tools, max_steps):
    from scaffold.tool_schema import validate

    items = list(script or [])[:max(0, int(max_steps))]
    tools = tools or {}
    trace, final, why = [], None, "budget"
    for i, reply in enumerate(items, 1):
        reply = reply or {}
        thought = reply.get("thought", "")
        if "final" in reply:
            final = reply["final"]
            trace.append({"step": i, "type": "final", "thought": thought, "final": final})
            why = "final"
            break
        action = reply.get("action")
        if not action:
            trace.append({"step": i, "type": "noop", "thought": thought})
            continue
        name = action.get("tool", "")
        args = action.get("args", {}) or {}
        valid = name in tools and validate(tools[name], args)["ok"]
        trace.append({
            "step": i, "type": "action", "thought": thought,
            "tool": name, "args": args, "valid": bool(valid),
        })
    else:
        why = "budget" if len(script or []) >= max(0, int(max_steps)) else "script_end"
    return {"trace": trace, "steps_used": len(trace), "terminated_by": why, "final": final}
