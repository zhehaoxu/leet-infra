"""Step 01 reference: deterministic tool-schema validation.

``validate`` never raises on malformed input: it reports every problem at once
so the caller can build a complete error message. It is a pure function of
``(spec, args)`` -- the LLM that produced the call is *not* part of the judge.
"""

_TYPE_HINTS = ("any", "str", "int", "float", "number", "bool", "list", "dict")


def _type_ok(value, kind):
    if kind in ("any", None, ""):
        return True
    if kind == "str":
        return isinstance(value, str)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "list":
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    return False


def validate(spec, args):
    """Validate ``args`` against ``spec`` and return a normalized call.

    spec = {"name": str, "strict": bool,
            "params": {name: {"type": str, "required": bool, "default": ...}}}
    Returns {"ok", "tool", "missing", "unexpected", "type_errors", "normalized"}.
    Lists are sorted so the result is stable across Python hash seeds.
    """
    params = spec.get("params", {}) or {}
    args = args or {}
    strict = bool(spec.get("strict", True))
    missing, type_errors, normalized = [], [], {}
    for name in sorted(params):
        rule = params[name] or {}
        if name in args:
            value = args[name]
            if _type_ok(value, rule.get("type", "any")):
                normalized[name] = value
            else:
                type_errors.append(name)
        elif "default" in rule:
            normalized[name] = rule["default"]
        elif rule.get("required", False):
            missing.append(name)
    unexpected = sorted(k for k in args if k not in params) if strict else []
    ok = (
        bool(spec.get("name"))
        and not missing
        and not unexpected
        and not type_errors
    )
    return {
        "ok": ok,
        "tool": spec.get("name", ""),
        "missing": missing,
        "unexpected": unexpected,
        "type_errors": type_errors,
        "normalized": normalized,
    }
