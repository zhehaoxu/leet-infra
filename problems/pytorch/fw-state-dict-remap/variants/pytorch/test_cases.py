import torch

# ---- 旧 checkpoint 的确定性构造（与题面数值规则一致，oracle 直接由数列公式手推） ----
_CKPT_KEYS = [
    "transformer.h.0.w", "transformer.h.0.b", "transformer.h.0.ln",
    "transformer.h.1.w", "transformer.h.1.b", "transformer.h.1.ln",
    "embed.w", "lm_head.weight",
]
_CKPT_SHAPES = [(2, 3), (2,), (3,), (4, 2), (4,), (2,), (3, 2), (2, 2)]


def _build_ckpt_values():
    """独立重算入口：连续递增数列 cursor 分配，返回 key -> flat list 的期望值表。"""
    out, cursor = {}, 0
    for key, shape in zip(_CKPT_KEYS, _CKPT_SHAPES):
        n = 1
        for s in shape:
            n *= s
        out[key] = [float(cursor + i) for i in range(n)]
        cursor += n
    return out


def _oracle():
    """独立路径：不走 load_state_dict，纯 Python 集合运算推 missing/unexpected，
    手工按新命名拼装期望参数值（embed 做转置展开）。"""
    vals = _build_ckpt_values()
    mapping = {
        "transformer.h.0.w": "blocks.0.weight",
        "transformer.h.0.b": "blocks.0.bias",
        "transformer.h.0.ln": "blocks.0.layernorm",
        "transformer.h.1.w": "blocks.1.weight",
        "transformer.h.1.b": "blocks.1.bias",
        "transformer.h.1.ln": "blocks.1.layernorm",
        "embed.w": "embed.weight",
    }
    # 新模型全部参数名（题面固定结构）：embed + 2 个 block（weight/bias/layernorm/gate）
    new_keys = [
        "embed.weight",
        "blocks.0.weight", "blocks.0.bias", "blocks.0.layernorm", "blocks.0.gate",
        "blocks.1.weight", "blocks.1.bias", "blocks.1.layernorm", "blocks.1.gate",
    ]
    loaded_vals = {
        "blocks.0.weight": vals["transformer.h.0.w"],
        "blocks.0.bias": vals["transformer.h.0.b"],
        "blocks.0.layernorm": vals["transformer.h.0.ln"],
        "blocks.1.weight": vals["transformer.h.1.w"],
        "blocks.1.bias": vals["transformer.h.1.b"],
        "blocks.1.layernorm": vals["transformer.h.1.ln"],
        # embed.w 是 (3,2)：[[25,26],[27,28],[29,30]]，转置成 (2,3) 行主序展平
        "embed.weight": [25.0, 27.0, 29.0, 26.0, 28.0, 30.0],
        "blocks.0.gate": [0.0, 0.0],
        "blocks.1.gate": [0.0, 0.0, 0.0, 0.0],
    }
    consumed = set(mapping.values())
    missing = sorted(k for k in new_keys if k not in consumed)
    unexpected = sorted(k for k in vals if mapping.get(k, k) not in new_keys)
    return {
        "mapping": dict(sorted(mapping.items())),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loaded": loaded_vals,
    }


EXPECTED = _oracle()


def check_example(solve):
    out = solve()
    assert isinstance(out, dict), "必须返回题面约定的 dict"
    assert out["mapping"] == EXPECTED["mapping"], f"mapping 不符: {out['mapping']}"
    assert out["missing_keys"] == EXPECTED["missing_keys"], \
        f"missing_keys 不符: got {out['missing_keys']}, want {EXPECTED['missing_keys']}"
    assert out["unexpected_keys"] == EXPECTED["unexpected_keys"], \
        f"unexpected_keys 不符: got {out['unexpected_keys']}"
    assert set(out["loaded"].keys()) == set(EXPECTED["loaded"].keys()), \
        f"loaded 键集合不符: {sorted(out['loaded'].keys())}"
    for k, want in EXPECTED["loaded"].items():
        got = out["loaded"][k]
        assert len(got) == len(want), f"{k} 长度不符"
        assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), f"{k} 数值不符: {got}"


check_example.example = True


def check_matches_oracle(solve):
    out = solve()
    assert out["mapping"] == EXPECTED["mapping"]
    assert out["missing_keys"] == EXPECTED["missing_keys"]
    assert out["unexpected_keys"] == EXPECTED["unexpected_keys"]
    for k, want in EXPECTED["loaded"].items():
        assert torch.equal(torch.tensor(out["loaded"][k]), torch.tensor(want)), f"{k} 与 oracle 不符"


def check_gate_left_at_init(solve):
    # missing 的参数必须保持新模型的零初始化，不被任何旧值污染
    out = solve()
    assert out["loaded"]["blocks.0.gate"] == [0.0, 0.0], "gate 应保持零初始化"
    assert out["loaded"]["blocks.1.gate"] == [0.0] * 4, "gate 应保持零初始化"


def check_transpose_applied(solve):
    # embed.weight 必须是转置后的行主序（非原样拷贝）
    out = solve()
    flat = out["loaded"]["embed.weight"]
    assert flat != [25.0, 26.0, 27.0, 28.0, 29.0, 30.0], "embed 应转置存放，不能原样拷贝"
    m = torch.tensor(flat).reshape(2, 3)
    assert m[0, 0] == 25 and m[0, 1] == 27 and m[1, 0] == 26


def check_keys_are_sorted_lists(solve):
    out = solve()
    assert out["missing_keys"] == sorted(out["missing_keys"]), "missing_keys 必须排序返回"
    assert out["unexpected_keys"] == sorted(out["unexpected_keys"]), "unexpected_keys 必须排序返回"
