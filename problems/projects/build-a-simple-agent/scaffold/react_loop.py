"""Step 02 reference: a ReAct loop state machine driven by fixture replies.

Model output is an explicit argument (``script``): the judge never calls an LLM.
Each reply is either {"thought", "action": {...}} or {"thought", "final": ...}.
"""


def run_loop(script, tools, max_steps):
    """Consume at most ``max_steps`` fixture replies and return the trace.

    The loop stops on the first reply carrying ``final``. ``terminated_by`` is
    one of "final" / "budget" / "script_end". Actions are schema-checked with
    the frozen step-01 module; invalid actions are recorded but never executed.
    """
    from scaffold.tool_schema import validate

    script = list(script or [])
    tools = tools or {}
    limit = max(0, int(max_steps))
    trace = []
    final = None
    why = "budget"
    window = script[:limit]
    for index, reply in enumerate(window, 1):
        thought = (reply or {}).get("thought", "")
        if "final" in (reply or {}):
            final = reply["final"]
            trace.append(
                {"step": index, "type": "final", "thought": thought, "final": final}
            )
            why = "final"
            break
        action = (reply or {}).get("action")
        if not action:
            trace.append({"step": index, "type": "noop", "thought": thought})
            continue
        name = action.get("tool", "")
        args = action.get("args", {}) or {}
        spec = tools.get(name)
        valid = spec is not None and validate(spec, args)["ok"]
        trace.append(
            {
                "step": index,
                "type": "action",
                "thought": thought,
                "tool": name,
                "args": args,
                "valid": bool(valid),
            }
        )
    else:
        why = "budget" if len(script) >= limit else "script_end"
    return {
        "trace": trace,
        "steps_used": len(trace),
        "terminated_by": why,
        "final": final,
    }
