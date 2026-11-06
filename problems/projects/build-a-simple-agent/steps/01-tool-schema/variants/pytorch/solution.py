"""Reference: deterministic tool-schema validation (step 01)."""


def solve(spec, args):
    params = spec.get("params", {}) or {}
    args = args or {}
    strict = bool(spec.get("strict", True))
    missing, bad_types, normalized = [], [], {}
    for name in sorted(params):
        rule = params[name] or {}
        kind = rule.get("type", "any")
        if name in args:
            value = args[name]
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            checks = {
                "any": True, "str": isinstance(value, str),
                "int": isinstance(value, int) and not isinstance(value, bool),
                "float": numeric, "number": numeric,
                "bool": isinstance(value, bool), "list": isinstance(value, list),
                "dict": isinstance(value, dict),
            }
            if checks.get(kind, False):
                normalized[name] = value
            else:
                bad_types.append(name)
        elif "default" in rule:
            normalized[name] = rule["default"]
        elif rule.get("required", False):
            missing.append(name)
    extra = sorted(k for k in args if k not in params) if strict else []
    return {
        "ok": bool(spec.get("name")) and not missing and not extra and not bad_types,
        "tool": spec.get("name", ""),
        "missing": missing,
        "unexpected": extra,
        "type_errors": bad_types,
        "normalized": normalized,
    }
