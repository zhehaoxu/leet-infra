import io
import pickletools
import zipfile

import torch
import torch.nn as nn
from collections import Counter, OrderedDict


def solve():
    """torch.save/weights_only load 白名单账 + load_state_dict 键级账。
    手写判定层：torch.save 打 zip 包，抽出 archive/data.pkl 后用
    pickletools 扫 GLOBAL 指令（protocol 2 全部为两行文本的 GLOBAL，
    无 STACK_GLOBAL），逐条对照题面钉死的允许键表（从官方
    _weights_only_unpickler._get_allowed_globals 在 protocol-2 序列化下
    实际触发的键反推）。不得调用 torch.load / Module.load_state_dict。

    官方口径（torch/serialization.py + _weights_only_unpickler.py v2.14.1）：
      - torch.load 自 2.6 起 weights_only **默认 True**：pkl 内任何
        GLOBAL 不在白名单 -> UnpicklingError（消息前缀
        "Weights only load failed"）
      - 纯 primitive 容器（dict/list/tuple/int/float/str/None/bool）不产生
        GLOBAL 指令 -> 天然放行
      - 白名单（protocol-2 键形态，空格分隔 module name）：
          collections OrderedDict / collections Counter
          torch Size / torch device / torch float32（dtype 键）
          torch FloatStorage（等各 dtype storage 类）
          torch._utils _rebuild_tensor_v2 / _rebuild_parameter
          __builtin__ set / __builtin__ bytearray / __builtin__ complex
          _codecs encode（bytes 协助）
      - 典型拒绝：torch.nn.modules.linear Linear（nn.Module）、
        types SimpleNamespace、__builtin__ xrange（range 对象！）、
        numpy dtype / numpy ndarray / numpy._core multiarray _reconstruct
      - lambda 在 torch.save 阶段即 PicklingError（save_fail，非 load 拒）
      - load_state_dict：missing = 模型有而 state_dict 无；unexpected =
        state_dict 有而模型无；两者**sorted 后**报出；strict=True 且任一
        非空 -> RuntimeError（消息含 "Error(s) in loading state_dict"）；
        strict=False -> IncompatibleKeys；全匹配 -> "All keys matched..."

    返回 dict（键见测试断言）：
      allowed_globals   sorted 的允许键表
      verdicts          {场景名: "ok" | "rejected" | "save_fail"}
      sd_*              load_state_dict 账
    """
    ALLOWED = {
        "collections OrderedDict", "collections Counter",
        "torch Size", "torch device", "torch float32",
        "torch FloatStorage", "torch LongStorage", "torch ByteStorage",
        "torch DoubleStorage",
        "torch._utils _rebuild_tensor_v2", "torch._utils _rebuild_parameter",
        "__builtin__ set", "__builtin__ bytearray", "__builtin__ complex",
        "_codecs encode",
    }

    cases = [
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

    verdicts = {}
    for name, obj in cases:
        buf = io.BytesIO()
        try:
            torch.save(obj, buf)
        except Exception:
            verdicts[name] = "save_fail"
            continue
        zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
        pkl = zf.read("archive/data.pkl")
        globals_seen = set()
        for op, arg, _pos in pickletools.genops(io.BytesIO(pkl)):
            if op.name in ("GLOBAL", "STACK_GLOBAL"):
                globals_seen.add(str(arg))
        verdicts[name] = ("ok" if all(g in ALLOWED for g in globals_seen)
                          else "rejected")

    # load_state_dict 键级账（手写：sorted 差集；注意 nn.Linear 根模块
    # 的 state_dict 键是 weight/bias，无属性名前缀）
    model_keys = ["weight", "bias"]
    other_keys = ["0.weight", "0.bias"]
    def ledger(model_keys, sd_keys):
        missing = sorted(set(model_keys) - set(sd_keys))
        unexpected = sorted(set(sd_keys) - set(model_keys))
        return missing, unexpected
    m_all, u_all = ledger(model_keys, ["weight", "bias"])
    m_seq, u_seq = ledger(other_keys, ["weight", "bias"])
    m_ext, u_ext = ledger(other_keys, ["weight", "bias", "extra.weight"])

    return {
        "allowed_globals": sorted(ALLOWED),
        "verdicts": verdicts,
        "sd_full": {"missing": m_all, "unexpected": u_all},
        "sd_foreign": {"missing": m_seq, "unexpected": u_seq},
        "sd_extra": {"missing": m_ext, "unexpected": u_ext},
    }
