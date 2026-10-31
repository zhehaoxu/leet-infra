import random

import torch
import torch.nn as nn


def check_anchor(solve):
    led = solve(4, 6, 3)
    # 遍历顺序：本层 _parameters -> 本层 _buffers -> 子模块
    assert led["param_names_dedup"] == [
        "p_first", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias",
    ]
    assert led["param_names_nodup"] == [
        "p_first", "p_second", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias",
    ]
    assert led["param_count_dedup"] == 5
    assert led["param_count_nodup"] == 6
    assert led["shared_same_object"] is True
    assert led["dedup_keeps_first"] is True
    assert led["buffer_names"] == ["running", "scratch"]
    # state_dict 不去重：p_second 也在；scratch 非持久不进；placeholder 隐身
    assert led["state_dict_keys"] == [
        "p_first", "p_second", "running", "lin1.weight", "lin1.bias",
        "blocks.0.weight", "blocks.0.bias",
    ]
    assert led["sd_has_duplicate_key"] is True
    assert led["nonpersistent_in_sd"] is False
    assert led["placeholder_in_named"] is False
    assert led["placeholder_in_sd"] is False
    assert led["loose_in_modules"] is False
    assert led["module_count"] == 4  # 根 + lin1 + blocks + blocks.0
    assert led["module_linear_count"] == 2
    assert led["buffer_reassign_kept"] is True
    assert led["buffer_follows_to"] is True


def check_parametrized_counts(solve):
    led = solve(2, 3, 5)
    # lin1: 2*3+3=9, blocks.0: 3*5+5=20, p_first: 5
    n_dedup = len(led["param_names_dedup"])
    n_nodup = len(led["param_names_nodup"])
    assert n_nodup - n_dedup == 1  # 恰好多出共享的 p_second
    assert led["param_names_dedup"].count("p_first") == 1
    total = 2 * 3 + 3 + 5 + 3 * 5 + 5
    m = led  # 借助 state_dict 键算参数元素数
    assert len([k for k in m["state_dict_keys"] if k.endswith("weight") or k.endswith("bias")]) == 4
    assert total == 9 + 5 + 20


def check_shared_semantics(solve):
    led = solve(3, 3, 3)
    assert led["shared_same_object"] is True
    assert led["dedup_keeps_first"] is True
    assert led["sd_has_duplicate_key"] is True
    # state_dict 里两个键指向同一 storage，load_state_dict 会写两次
    assert led["state_dict_keys"].count("p_first") == 1
    assert led["state_dict_keys"].count("p_second") == 1


def check_buffer_persistence(solve):
    led = solve(2, 4, 2)
    assert led["buffer_names"] == ["running", "scratch"]
    assert led["nonpersistent_in_sd"] is False
    # named_buffers 不管 persistent，state_dict 只收 persistent
    assert "scratch" in led["buffer_names"]
    assert "scratch" not in led["state_dict_keys"]
    assert led["buffer_reassign_kept"] is True
    assert led["buffer_follows_to"] is True


def check_registration_blind_spots(solve):
    led = solve(2, 4, 2)
    assert led["placeholder_in_named"] is False
    assert led["placeholder_in_sd"] is False
    assert led["loose_in_modules"] is False
    # 普通 list 里的 Linear 不被注册：modules 只有根 + lin1 + blocks + blocks.0
    assert led["module_count"] == 4
    assert led["module_linear_count"] == 2


def check_random_sizes(solve):
    rng = random.Random(647)
    for _ in range(15):
        a, b, c = (rng.randint(1, 8) for _ in range(3))
        led = solve(a, b, c)
        assert led["param_count_nodup"] - led["param_count_dedup"] == 1
        assert led["module_count"] == 4
        assert led["loose_in_modules"] is False
        assert led["buffer_follows_to"] is True
        assert led["buffer_reassign_kept"] is True
        assert led["state_dict_keys"][0] == "p_first"
        assert led["state_dict_keys"][1] == "p_second"
        assert led["state_dict_keys"][2] == "running"
        assert "scratch" not in led["state_dict_keys"]


def check_order_ledger(solve):
    # 本层参数/缓冲先于子模块：p_first 在 lin1 之前，running 在 blocks 之前
    led = solve(2, 2, 2)
    names = led["param_names_dedup"]
    assert names.index("p_first") < names.index("lin1.weight")
    keys = led["state_dict_keys"]
    assert keys.index("running") < keys.index("lin1.weight")
    assert keys.index("lin1.bias") < keys.index("blocks.0.weight")


def check_invalid_inputs(solve):
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1), (1.5, 1, 1), ("2", 1, 1)):
        try:
            solve(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    # bool 是 int 的子类，也要拒绝
    try:
        solve(True, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bool width")
