"""参考答案：旧命名 checkpoint 重映射到新模块，strict=False 加载并记全 missing/unexpected 账。"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.layernorm = nn.Parameter(torch.zeros(in_f))
        self.gate = nn.Parameter(torch.zeros(out_f))


class Embed(nn.Module):
    def __init__(self, out_f: int, in_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_f, in_f))


class NewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embed(2, 3)
        self.blocks = nn.ModuleList([Block(3, 2), Block(2, 4)])


def build_old_ckpt() -> dict:
    """按固定顺序枚举旧键，值取连续递增的确定性数列（不依赖随机数）。"""
    sd: dict[str, torch.Tensor] = {}
    cursor = 0

    def seq(*shape: int) -> torch.Tensor:
        nonlocal cursor
        n = 1
        for s in shape:
            n *= s
        vals = torch.arange(n, dtype=torch.float32).reshape(shape) + cursor
        cursor += n
        return vals

    sd["transformer.h.0.w"] = seq(2, 3)
    sd["transformer.h.0.b"] = seq(2)
    sd["transformer.h.0.ln"] = seq(3)
    sd["transformer.h.1.w"] = seq(4, 2)
    sd["transformer.h.1.b"] = seq(4)
    sd["transformer.h.1.ln"] = seq(2)
    sd["embed.w"] = seq(3, 2)          # 旧 Conv1D 风格：(out, in) 相对新 (2,3) 转置存放
    sd["lm_head.weight"] = seq(2, 2)   # 新模型没有 lm_head → unexpected
    return sd


def remap_key(key: str) -> str:
    if key.startswith("transformer.h."):
        rest = key[len("transformer.h."):]
        idx, _, suffix = rest.partition(".")
        new_suffix = {"w": "weight", "b": "bias", "ln": "layernorm"}[suffix]
        return f"blocks.{idx}.{new_suffix}"
    if key == "embed.w":
        return "embed.weight"
    return key


def solve():
    ckpt = build_old_ckpt()
    new_sd: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}
    for old_key, tensor in ckpt.items():
        new_key = remap_key(old_key)
        if new_key != old_key:
            mapping[old_key] = new_key
        if new_key == "embed.weight":
            tensor = tensor.T.contiguous()  # 值变换：转置对齐新布局
        new_sd[new_key] = tensor

    model = NewNet()
    incompatible = model.load_state_dict(new_sd, strict=False)

    loaded = {
        name: param.detach().reshape(-1).tolist()
        for name, param in model.state_dict().items()
    }
    return {
        "mapping": dict(sorted(mapping.items())),
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
        "loaded": loaded,
    }
