import io

import torch
import torch.nn as nn
from collections import Counter, OrderedDict


def _save_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _wo_load(b):
    """真 torch.load(weights_only=True)；返回 (ok, exc_type_name)。"""
    try:
        torch.load(io.BytesIO(b), weights_only=True)
        return True, None
    except Exception as e:
        return False, type(e).__name__


def _cases():
    return [
        ("tensor_dict", {"w": torch.tensor([1.0, 2.0])}),
        ("primitives", {"n": None, "i": 3, "s": "x", "f": 0.5, "b": False}),
        ("nested", {"d": {"a": [1, [2, {"b": (3, 4)}]]}}),
        ("ordered", OrderedDict([("a", 1)])),
        ("counter", Counter({"x": 1})),
        ("size", {"sz": torch.Size([2, 3])}),
        ("device", {"dev": torch.device("cpu")}),
        ("dtype", {"dt": torch.float32}),
        ("set", {"st": {1, 2, 3}}),
        ("bytearray", {"ba": bytearray(b"ab")}),
        ("complex", {"cx": complex(1, 2)}),
        ("param", {"p": nn.Parameter(torch.zeros(2))}),
        ("linear", {"m": nn.Linear(2, 2)}),
        ("namespace", {"ns": __import__("types").SimpleNamespace(a=1)}),
        ("lambda_fn", {"l": lambda x: x}),
        ("range_obj", {"r": range(5)}),
        ("np_array", {"a": __import__("numpy").zeros((2, 2))}),
    ]


def _py():
    """oracle：官方 torch.load(weights_only=True) 逐场景真跑出 verdict；
    值 roundtrip 对 ok 场景断言值全等；load_state_dict 用真 Module。"""
    verdicts = {}
    for name, obj in _cases():
        try:
            b = _save_bytes(obj)
        except Exception:
            verdicts[name] = "save_fail"
            continue
        ok, _ = _wo_load(b)
        verdicts[name] = "ok" if ok else "rejected"

    # ok 场景值 roundtrip：weights_only load 回来的值逐项一致
    r = torch.load(io.BytesIO(_save_bytes({"w": torch.tensor([1.0, 2.0])})),
                   weights_only=True)
    assert torch.equal(r["w"], torch.tensor([1.0, 2.0]))
    r2 = torch.load(io.BytesIO(_save_bytes(OrderedDict([("a", 1)]))),
                    weights_only=True)
    assert list(r2.items()) == [("a", 1)]

    # 官方默认 weights_only（2.6+ = True）：裸 load nn.Linear 同样被拒
    try:
        torch.load(io.BytesIO(_save_bytes({"m": nn.Linear(2, 2)})))
        default_rejects = False
    except Exception:
        default_rejects = True
    # 裸 load 纯 tensor dict 仍 ok
    torch.load(io.BytesIO(_save_bytes({"w": torch.tensor([1.0])})))
    default_ok = True

    # load_state_dict 真账
    m1 = nn.Linear(2, 3)
    r_match = m1.load_state_dict(m1.state_dict())
    m2 = nn.Sequential(nn.Linear(2, 3))
    r_seq = m2.load_state_dict(m1.state_dict(), strict=False)
    try:
        m2.load_state_dict(m1.state_dict())
        strict_err = None
    except RuntimeError as e:
        strict_err = str(e)
    sd_extra = dict(m1.state_dict())
    sd_extra["extra.weight"] = torch.zeros(1)
    r_ext = m2.load_state_dict(sd_extra, strict=False)
    return {
        "verdicts": verdicts,
        "default_rejects": default_rejects,
        "default_ok": default_ok,
        "sd_match": repr(r_match) == "<All keys matched successfully>",
        "sd_foreign": {"missing": sorted(r_seq.missing_keys),
                       "unexpected": sorted(r_seq.unexpected_keys)},
        "sd_extra": {"missing": sorted(r_ext.missing_keys),
                     "unexpected": sorted(r_ext.unexpected_keys)},
        "strict_err_prefix": strict_err.splitlines()[0]
        if strict_err else None,
    }


def _compare(solve):
    led = solve()
    ref = _py()
    assert led["verdicts"] == ref["verdicts"], (led["verdicts"], ref["verdicts"])
    assert led["sd_foreign"] == ref["sd_foreign"], (led["sd_foreign"],
                                                    ref["sd_foreign"])
    assert led["sd_extra"] == ref["sd_extra"]
    assert led["sd_full"] == {"missing": [], "unexpected": []}
    assert ref["sd_match"] and ref["default_rejects"] and ref["default_ok"]
    assert ref["strict_err_prefix"].startswith(
        "Error(s) in loading state_dict")
    return led


def check_verdicts(solve):
    led = _compare(solve)
    v = led["verdicts"]
    ok_group = ["tensor_dict", "primitives", "nested", "ordered", "counter",
                "size", "device", "dtype", "set", "bytearray", "complex",
                "param"]
    for k in ok_group:
        assert v[k] == "ok", (k, v[k])
    for k in ("linear", "namespace", "range_obj", "np_array"):
        assert v[k] == "rejected", (k, v[k])
    assert v["lambda_fn"] == "save_fail"


def check_allowed_table(solve):
    led = _compare(solve)
    # 钉死表是 sorted 的，且必须包含张量重建核心键
    ag = led["allowed_globals"]
    assert ag == sorted(ag)
    for g in ("torch._utils _rebuild_tensor_v2", "torch._utils _rebuild_parameter",
              "collections OrderedDict", "torch FloatStorage",
              "__builtin__ set", "_codecs encode"):
        assert g in ag, g


def check_state_dict_ledger(solve):
    led = _compare(solve)
    # nn.Linear 根模块键为 weight/bias（无属性名前缀）；sorted 字典序
    assert led["sd_foreign"] == {"missing": ["0.bias", "0.weight"],
                                 "unexpected": ["bias", "weight"]}
    assert led["sd_extra"]["unexpected"] == ["bias", "extra.weight", "weight"]
    assert led["sd_extra"]["missing"] == ["0.bias", "0.weight"]


def check_invalid_no_torch_load(solve):
    """红线：solution 源码不得出现 torch.load / load_state_dict 调用
    （selftest harness 的模块无 __file__ 时跳过——变异 harness 与
    integrity 护栏兜底）。"""
    import os
    import sys
    mod = sys.modules.get(__name__)
    src = None
    if mod is not None and getattr(mod, "__file__", None):
        src = open(os.path.join(os.path.dirname(mod.__file__),
                                "solution.py")).read()
    if src is None:
        return
    assert "torch.load(" not in src
    assert ".load_state_dict(" not in src
