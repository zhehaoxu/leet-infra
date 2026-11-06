"""L1 oracle comparison + L2 invariants for step 01."""

SPEC = {
    "name": "search",
    "strict": True,
    "params": {
        "query": {"type": "str", "required": True},
        "top_k": {"type": "int", "required": False, "default": 3},
        "alpha": {"type": "number", "required": False, "default": 0.5},
    },
}


def _oracle(spec, args):
    """Independent oracle: filter-first, then classify each declared key."""
    declared = set(spec.get("params", {}))
    present = {k: v for k, v in args.items() if k in declared}
    wanted = sorted(spec.get("params", {}))
    required = [k for k in wanted if spec["params"][k].get("required", False)]
    types = {
        "str": lambda v: type(v) is str,
        "int": lambda v: type(v) is int,
        "float": lambda v: type(v) in (int, float),
        "number": lambda v: type(v) in (int, float),
        "bool": lambda v: type(v) is bool,
        "list": lambda v: type(v) is list,
        "dict": lambda v: type(v) is dict,
        "any": lambda v: True,
    }
    wrong = [k for k, v in present.items()
             if not types.get(spec["params"][k].get("type", "any"), lambda v: False)(v)]
    normalized = {k: present[k] for k in wanted if k in present and k not in wrong}
    for k in wanted:
        if k not in present and "default" in spec["params"][k]:
            normalized[k] = spec["params"][k]["default"]
    missing = [k for k in required if k not in present]
    extra = sorted(set(args) - declared) if spec.get("strict", True) else []
    return {
        "ok": bool(spec.get("name")) and not (missing or extra or wrong),
        "tool": spec.get("name", ""),
        "missing": missing,
        "unexpected": extra,
        "type_errors": sorted(wrong),
        "normalized": normalized,
    }


def _compare(solve, spec, args):
    got = solve(spec, args)
    assert got == _oracle(spec, args), f"oracle mismatch: {got}"
    return got


def check_basic(solve):
    out = _compare(solve, SPEC, {"query": "hello", "top_k": 5})
    assert out["ok"] is True
    assert out["normalized"] == {"query": "hello", "top_k": 5, "alpha": 0.5}


def check_missing_and_unexpected(solve):
    out = _compare(solve, SPEC, {"top_k": 2, "junk": 1})
    assert out["missing"] == ["query"]
    assert out["unexpected"] == ["junk"]
    assert out["ok"] is False
    loose = dict(SPEC, strict=False)
    assert _compare(solve, loose, {"query": "x", "junk": 1})["unexpected"] == []


def check_type_errors(solve):
    out = _compare(solve, SPEC, {"query": True, "top_k": "3", "alpha": "bad"})
    assert out["type_errors"] == ["alpha", "query", "top_k"]
    assert out["normalized"] == {}


check_basic.example = True


def check_property_idempotent(solve):
    """L2: normalizing an already-normalized call is a fixed point."""
    first = solve(SPEC, {"query": "abc"})
    second = solve(SPEC, first["normalized"])
    assert first == second
    assert solve(SPEC, {"query": "abc"}) == first
    assert sorted(first["normalized"]) == sorted(SPEC["params"])


def check_property_bounded(solve):
    """L2: no unexpected key ever leaks into the normalized call."""
    for args in ({}, {"query": "x", "a": 1, "b": 2}, {"query": "x", "top_k": None}):
        out = _compare(solve, SPEC, args)
        assert set(out["normalized"]) <= set(SPEC["params"])
