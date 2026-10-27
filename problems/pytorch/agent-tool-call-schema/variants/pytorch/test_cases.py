"""独立 oracle：类型谓词表 + 递归路径拼接，与参考实现的分支写法互不共享代码。"""


def _tname(value):
    return "null" if value is None else type(value).__name__


def _p_string(v):
    return (True, v) if type(v) is str else (False, None)


def _p_boolean(v):
    return (True, v) if type(v) is bool else (False, None)


def _p_integer(v):
    if type(v) is int:
        return True, v
    if type(v) is float and v == int(v):
        return True, int(v)
    return False, None


def _p_number(v):
    return (True, float(v)) if type(v) in (int, float) else (False, None)


def _p_array(v):
    return (True, v) if type(v) is list else (False, None)


def _p_object(v):
    return (True, v) if type(v) is dict else (False, None)


_PREDICATES = {
    "string": _p_string,
    "boolean": _p_boolean,
    "integer": _p_integer,
    "number": _p_number,
    "array": _p_array,
    "object": _p_object,
}


def _oracle_validate(value, spec, path, errors):
    kind = spec.get("type")
    if kind is None:
        return True, value
    ok, val = _PREDICATES[kind](value)
    if not ok:
        errors.append("{}: expected {}, got {}".format(path, kind, _tname(value)))
        return False, None

    if "enum" in spec:
        hit = [e for e in spec["enum"] if type(e) is type(val) and e == val]
        if not hit:
            errors.append("{}: value {!r} is not one of {!r}".format(path, val, list(spec["enum"])))
            return False, None

    if kind == "array" and "items" in spec:
        collected, all_good = [], True
        for i, item in enumerate(val):
            good, sub = _oracle_validate(item, spec["items"], "{}[{}]".format(path, i), errors)
            all_good = all_good and good
            collected.append(sub)
        if not all_good:
            return False, None
        val = collected

    if kind == "object" and ("properties" in spec or "required" in spec):
        built, sub_errors = _oracle_object(val, spec, path)
        if sub_errors:
            errors.extend(sub_errors)
            return False, None
        val = built

    return True, val


def _oracle_object(args, schema, prefix):
    props = schema.get("properties", {})
    result, problems = {}, []

    for name in schema.get("required", []):
        if name in args:
            continue
        if name in props and "default" in props[name]:
            continue
        problems.append("{}: missing required property".format(_join(prefix, name)))

    for name, spec in props.items():
        if name not in args:
            if "default" in spec:
                result[name] = spec["default"]
            continue
        good, val = _oracle_validate(args[name], spec, _join(prefix, name), problems)
        if good:
            result[name] = val

    for name in [k for k in args if k not in props]:
        if schema.get("additionalProperties") is False:
            problems.append("{}: unexpected property".format(_join(prefix, name)))
        else:
            result[name] = args[name]

    return result, problems


def _join(prefix, name):
    return prefix + "." + name


def _oracle(schema, arguments):
    return _oracle_object(arguments, schema, "$")


_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "default": 1},
        "unit": {"type": "string", "enum": ["c", "f"]},
    },
    "required": ["city", "unit"],
}


def check_example(solve):
    called, errors = solve(_WEATHER_SCHEMA, {"city": "Beijing", "unit": "c", "days": 3.0})
    assert errors == [], errors
    assert called == {"city": "Beijing", "days": 3, "unit": "c"}, called
    assert type(called["days"]) is int, "integer 应归一成 int"


check_example.example = True


def check_required_and_defaults(solve):
    called, errors = solve(_WEATHER_SCHEMA, {"city": "Beijing"})
    assert errors == ["$.unit: missing required property"], errors
    # unit 校验失败不写入，但默认值照填
    assert called == {"city": "Beijing", "days": 1}, called

    # required 字段带 default：按补齐处理，不报缺失
    schema = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "default": 10}},
        "required": ["limit"],
    }
    called, errors = solve(schema, {})
    assert (called, errors) == ({"limit": 10}, []), (called, errors)


def check_type_rules(solve):
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "flag": {"type": "boolean"},
            "name": {"type": "string"},
        },
        "required": ["n", "x", "flag", "name"],
    }
    called, errors = solve(schema, {"n": 3.0, "x": 2, "flag": True, "name": "a"})
    assert errors == [], errors
    assert type(called["n"]) is int and called["n"] == 3
    assert type(called["x"]) is float and called["x"] == 2.0

    called, errors = solve(schema, {"n": 3.5, "x": True, "flag": 1, "name": 5})
    assert errors == [
        "$.n: expected integer, got float",
        "$.x: expected number, got bool",
        "$.flag: expected boolean, got int",
        "$.name: expected string, got int",
    ], errors
    assert called == {}, called

    # bool 是 int 的子类：integer 必须显式拒绝它
    assert solve({"type": "object", "properties": {"n": {"type": "integer"}}}, {"n": True})[1] == [
        "$.n: expected integer, got bool"
    ]


def check_enum(solve):
    called, errors = solve(_WEATHER_SCHEMA, {"city": "Beijing", "unit": "k"})
    assert errors == ["$.unit: value 'k' is not one of ['c', 'f']"], errors
    assert "unit" not in called

    # 枚举在类型之后：bool 先被 integer 拒掉，值域外才轮到枚举报错
    schema = {"type": "object", "properties": {"k": {"type": "integer", "enum": [0, 1]}}}
    assert solve(schema, {"k": 1}) == ({"k": 1}, [])
    assert solve(schema, {"k": True})[1] == ["$.k: expected integer, got bool"]
    assert solve(schema, {"k": 2})[1] == ["$.k: value 2 is not one of [0, 1]"]

    # 比对是双重的：number 归一成 float 后不会命中 int 枚举
    num = {"type": "object", "properties": {"k": {"type": "number", "enum": [0, 1]}}}
    assert solve(num, {"k": 0})[1] == ["$.k: value 0.0 is not one of [0, 1]"]
    assert solve(num, {"k": 0.5})[1] == ["$.k: value 0.5 is not one of [0, 1]"]


def check_nested_extra_and_invalid(solve):
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "enum": [1, 5, 10]},
            "filters": {
                "type": "object",
                "properties": {"lang": {"type": "string"}, "year": {"type": "integer"}},
                "required": ["lang"],
            },
            "ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    }
    args = {"query": "kv cache", "ids": [1, 2.0, "3"], "filters": {"year": 2024}, "debug": True}

    called, errors = solve(schema, args)
    want_called, want_errors = _oracle(schema, args)
    assert errors == want_errors, f"{errors} != {want_errors}"
    assert called == want_called, f"{called} != {want_called}"
    assert errors == [
        "$.filters.lang: missing required property",
        "$.ids[2]: expected integer, got str",
        "$.debug: unexpected property",
    ], errors

    # 合法的嵌套与数组：逐元素归一（2.0 -> 2）
    ok_called, ok_errors = solve(schema, {"query": "q", "ids": [1, 2.0], "filters": {"lang": "zh"}})
    assert (ok_called, ok_errors) == ({"query": "q", "top_k": 5, "filters": {"lang": "zh"}, "ids": [1, 2]}, [])

    # additionalProperties 置 True 时多余字段原样保留
    loose = dict(schema)
    loose["additionalProperties"] = True
    kept, _ = solve(loose, {"query": "q", "debug": True})
    assert kept.get("debug") is True

    for bad in (
        lambda: solve("schema", {}),
        lambda: solve(schema, [("query", "q")]),
        lambda: solve({"properties": []}, {}),
        lambda: solve({"required": "query"}, {}),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
