_FAIL = object()

_NAMES = {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}


def _type_name(value):
    if value is None:
        return "null"
    return _NAMES.get(type(value), type(value).__name__)


def _coerce(want, value):
    """按目标类型归一，返回 _FAIL 表示类型不符。bool 一律先排除（它是 int 的子类）。"""
    if want == "string":
        return value if isinstance(value, str) else _FAIL
    if want == "boolean":
        return value if isinstance(value, bool) else _FAIL
    if want == "integer":
        if isinstance(value, bool):
            return _FAIL
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return _FAIL
    if want == "number":
        if isinstance(value, bool):
            return _FAIL
        if isinstance(value, (int, float)):
            return float(value)
        return _FAIL
    if want == "array":
        return value if isinstance(value, list) else _FAIL
    if want == "object":
        return value if isinstance(value, dict) else _FAIL
    return value  # 未知 type：不校验


def _in_enum(value, enum):
    """枚举按「类型 + 值」双重要求：True 与 1、1 与 1.0 都不互相顶替。"""
    for item in enum:
        if type(item) is type(value) and item == value:
            return True
    return False


def _validate(value, spec, path, errors):
    want = spec.get("type")
    if want is None:
        return True, value
    coerced = _coerce(want, value)
    if coerced is _FAIL:
        errors.append(f"{path}: expected {want}, got {_type_name(value)}")
        return False, None

    enum = spec.get("enum")
    if enum is not None and not _in_enum(coerced, enum):
        errors.append(f"{path}: value {coerced!r} is not one of {list(enum)!r}")
        return False, None

    if isinstance(coerced, list) and spec.get("items") is not None:
        items, ok = [], True
        for i, item in enumerate(coerced):
            good, val = _validate(item, spec["items"], f"{path}[{i}]", errors)
            ok = ok and good
            items.append(val)
        if not ok:
            return False, None
        coerced = items

    if isinstance(coerced, dict) and ("properties" in spec or "required" in spec):
        nested, sub_errors = _object(coerced, spec, path)
        errors.extend(sub_errors)
        if sub_errors:
            return False, None
        coerced = nested

    return True, coerced


def _object(arguments, schema, prefix):
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    out, errors = {}, []

    for name in required:
        if name in arguments:
            continue
        spec = props.get(name)
        if spec is not None and "default" in spec:
            continue  # 有默认值的 required 字段按补齐处理，不算缺失
        errors.append(f"{prefix}.{name}: missing required property")

    for name, spec in props.items():
        if name in arguments:
            good, val = _validate(arguments[name], spec, f"{prefix}.{name}", errors)
            if good:
                out[name] = val
        elif "default" in spec:
            out[name] = spec["default"]

    extras = [name for name in arguments if name not in props]
    if schema.get("additionalProperties") is False:
        errors.extend(f"{prefix}.{name}: unexpected property" for name in extras)
    else:
        for name in extras:
            out[name] = arguments[name]

    return out, errors


def solve(schema, arguments):
    if not isinstance(schema, dict):
        raise ValueError("schema 必须是 dict")
    if not isinstance(arguments, dict):
        raise ValueError("arguments 必须是 dict")
    props = schema.get("properties")
    if props is not None and not isinstance(props, dict):
        raise ValueError("schema['properties'] 必须是 dict")
    required = schema.get("required")
    if required is not None and not isinstance(required, list):
        raise ValueError("schema['required'] 必须是 list")
    return _object(arguments, schema, "$")
