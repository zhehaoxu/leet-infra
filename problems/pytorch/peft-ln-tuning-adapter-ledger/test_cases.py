"""order 834 判题用例：独立 oracle 用纯 Python 重写，不 import solution。

7 个 check_*：select / default_targets / update / trainable / forward /
merge_unmerge / unload。每个都做全网格对拍 + 手算锚点。
"""

import re

ORA_TARGETS = {
    "llama": ["input_layernorm", "post_attention_layernorm", "norm"],
    "bloom": ["input_layernorm", "post_attention_layernorm", "ln_f"],
    "t5": ["layer_norm", "final_layer_norm"],
    "gpt2": ["ln_1", "ln_2", "ln_f"],
    "llava": ["multi_modal_projector", "input_layernorm",
              "post_attention_layernorm", "norm", "embed_tokens", "lm_head"],
    "bart": ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"],
}


def ora_select(key, targets, exclude, mts):
    if exclude:
        if isinstance(exclude, str):
            hit = re.fullmatch(exclude, key) is not None
        else:
            hit = key in exclude or any(key.endswith("." + x) for x in exclude)
        if hit:
            return {"matched": False, "excluded": True,
                    "reason": "exclude_modules"}
    if mts:
        for m in mts:
            if re.match(r"(^|.*\.)%s($|\..*)" % m, key):
                return {"matched": False, "excluded": True,
                        "reason": "modules_to_save"}
    if isinstance(targets, str):
        ok = re.fullmatch(targets, key) is not None
    else:
        ok = (key in targets) or any(key.endswith("." + t) for t in targets)
    return {"matched": ok, "excluded": False,
            "reason": "ok" if ok else "no_match"}


def ora_default_targets(model_type):
    if model_type not in ORA_TARGETS:
        raise ValueError("unknown model_type: %s" % model_type)
    return {"targets": list(ORA_TARGETS[model_type]), "known": True}


def ora_update(weight, bias, adapter, active, inference, in_features):
    return {"original": {"weight": list(weight), "bias": list(bias)},
            "copy": {"weight": list(weight), "bias": list(bias)},
            "independent": True,
            "requires_grad": (not inference) and (adapter == active),
            "in_features": in_features, "out_features": in_features}


def ora_trainable(n, hidden, base, adapters, active, inference):
    p = n * 2 * hidden
    per = {}
    tr = 0
    for a in adapters:
        rg = (a == active) and (not inference)
        per[a] = {"requires_grad": rg, "params": p}
        if rg:
            tr += p
    return {"trainable_params": tr,
            "frozen_params": base + p * len(adapters) - tr,
            "per_adapter": per, "base_frozen": True}


def ora_forward(disable, merged, active, present, base_out, tuned_out):
    if disable:
        return {"branch": "disable_adapters", "out": base_out, "error": None,
                "unmerged_on_disable": merged}
    if merged or len(active) == 0:
        return {"branch": "base", "out": base_out, "error": None,
                "unmerged_on_disable": False}
    if len(active) != 1:
        return {"branch": "error", "out": None,
                "error": "multiple active adapters", "unmerged_on_disable": False}
    if not present:
        return {"branch": "adapter_missing", "out": base_out, "error": None,
                "unmerged_on_disable": False}
    return {"branch": "adapter", "out": tuned_out, "error": None,
            "unmerged_on_disable": False}


def ora_merge(action, names, active, present, merged, base_slot, adapter_slot):
    ns = list(names) if names is not None else list(active)
    m = list(merged)
    bs, ad = base_slot, adapter_slot
    warning = None
    error = None
    if action == "merge":
        if len(ns) > 1:
            error = "LN tuning does not allow merging more than one adapter"
        elif ns and ns[0] in present:
            if m:
                warning = "Already merged. Unmerging first."
                bs, ad = ad, bs
                m = []
            bs, ad = ad, bs
            m = [ns[0]]
    elif action == "unmerge":
        if not m:
            warning = "Already unmerged. Nothing to do."
        else:
            m = m[:-1]
            bs, ad = ad, bs
    return {"base_slot": bs, "adapter_slot": ad, "merged_adapters": m,
            "warning": warning, "error": error, "safe_merge_ignored": True}


def ora_unload(active, consider, mts, prefix, keys, with_base, merge):
    c = list(consider) if consider is not None else list(active)
    error = None
    if any(mts.get(a) for a in c) and len(c) > 1:
        error = "Cannot unload multiple adapters that specify modules_to_save."
    kl = [k for k in keys if prefix not in k]
    rep = [k for k in kl if k in set(with_base)] if error is None else []
    return {"error": error, "key_list": kl, "replaced": rep,
            "merge_calls": len(rep) if merge else 0}


def check_select(solve):
    keys = [
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
        "model.norm", "model.layers.0.self_attn.q_proj",
        "input_layernorm", "my_input_layernorm", "model.classifier",
        "model.classifier.bias", "model.layers.0.input_layernorm.weight",
    ]
    targets = [
        ["input_layernorm", "post_attention_layernorm", "norm"],
        ".*input_layernorm",
        ["norm"],
        [".*"],
        ".*norm",
    ]
    excludes = [None, ".*\\.q_proj", ["classifier"], ".*norm"]
    mts_list = [None, ["classifier"], ["layers"], ["norm"]]
    n = 0
    for key in keys:
        for tm in targets:
            for ex in excludes:
                for mts in mts_list:
                    got = solve(mode="select", key=key, target_modules=tm,
                                exclude_modules=ex, modules_to_save=mts)
                    exp = ora_select(key, tm, ex, mts)
                    assert got == exp, ("select %s tm=%r ex=%r mts=%r got=%r exp=%r"
                                        % (key, tm, ex, mts, got, exp))
                    n += 1
    # 手算锚点：list 后缀要求点、str 用 fullmatch、exclude 优先于 modules_to_save
    assert solve(mode="select", key="model.layers.0.input_layernorm",
                 target_modules=["input_layernorm"]) == {
        "matched": True, "excluded": False, "reason": "ok"}
    assert solve(mode="select", key="my_input_layernorm",
                 target_modules=["input_layernorm"]) == {
        "matched": False, "excluded": False, "reason": "no_match"}
    assert solve(mode="select", key="my_input_layernorm",
                 target_modules=".*input_layernorm") == {
        "matched": True, "excluded": False, "reason": "ok"}
    assert solve(mode="select", key="model.layers.0.self_attn.q_proj",
                 target_modules=".*", exclude_modules=".*\\.q_proj") == {
        "matched": False, "excluded": True, "reason": "exclude_modules"}
    assert solve(mode="select", key="model.classifier.bias",
                 target_modules=".*", modules_to_save=["classifier"]) == {
        "matched": False, "excluded": True, "reason": "modules_to_save"}
    assert solve(mode="select", key="model.layers.0.input_layernorm",
                 target_modules=".*", modules_to_save=["layers"]) == {
        "matched": False, "excluded": True, "reason": "modules_to_save"}
    assert solve(mode="select", key="model.classifier",
                 target_modules=".*", exclude_modules=["classifier"],
                 modules_to_save=["classifier"])["reason"] == "exclude_modules"
    print("select: %d grid + 7 anchors OK" % n)


def check_default_targets(solve):
    for name in ORA_TARGETS:
        got = solve(mode="default_targets", model_type=name)
        assert got == ora_default_targets(name), ("default_targets %s" % name)
    # 手算锚点：llama 3 项 / llava 6 项 / gpt2 3 项，顺序固定
    assert solve(mode="default_targets", model_type="llama")["targets"] == [
        "input_layernorm", "post_attention_layernorm", "norm"]
    assert len(solve(mode="default_targets", model_type="llava")["targets"]) == 6
    assert solve(mode="default_targets", model_type="gpt2")["targets"] == [
        "ln_1", "ln_2", "ln_f"]
    for bad in ("unknown_arch", "", "Llama"):
        try:
            solve(mode="default_targets", model_type=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown model_type must raise ValueError: %r" % bad)
    print("default_targets: 6 arch + 3 unknown OK")


def check_update(solve):
    n = 0
    for weight, bias in (([1.0, 2.0, 3.0], [0.5, -0.5, 0.0]),
                         ([-1.0], [2.0])):
        for adapter, active in (("a", "a"), ("a", "b"), ("b", "a")):
            for inference in (False, True):
                for in_features in (4, 16):
                    got = solve(mode="update", weight=weight, bias=bias,
                                adapter_name=adapter, active_adapter=active,
                                inference_mode=inference,
                                in_features=in_features)
                    exp = ora_update(weight, bias, adapter, active,
                                     inference, in_features)
                    for k in ("independent", "requires_grad", "in_features",
                              "out_features"):
                        assert got[k] == exp[k], ("update %s adapter=%s active=%s inf=%s: %r vs %r"
                                                  % (k, adapter, active, inference,
                                                     got[k], exp[k]))
                    assert got["original"] == exp["original"], "original values"
                    assert got["copy"] == exp["copy"], "copy values"
                    # deepcopy 独立性必须是「两个不同对象」，不是值相等
                    assert got["original"]["weight"] is not got["copy"]["weight"], \
                        "copy must be a distinct object from the base layer"
                    assert got["original"]["bias"] is not got["copy"]["bias"], \
                        "bias copy must be a distinct object"
                    n += 1
    # 手算锚点
    g = solve(mode="update", weight=[3.0, 4.0], bias=[1.0, 2.0],
              adapter_name="a", active_adapter="a", inference_mode=False,
              in_features=8)
    assert g["independent"] is True and g["requires_grad"] is True
    assert g["in_features"] == 8 and g["out_features"] == 8
    g2 = solve(mode="update", weight=[3.0, 4.0], bias=[1.0, 2.0],
               adapter_name="a", active_adapter="a", inference_mode=True,
               in_features=8)
    assert g2["requires_grad"] is False
    print("update: %d grid + 2 anchors OK" % n)


def check_trainable(solve):
    n = 0
    for n_ln in (2, 3):
        for hidden in (4, 8):
            for base in (0, 100):
                for adapters in (["default"], ["a", "b"]):
                    for active in ("default", "a"):
                        for inference in (False, True):
                            got = solve(mode="trainable", n_target_ln=n_ln,
                                        hidden=hidden, base_params=base,
                                        adapters=adapters,
                                        active_adapter=active,
                                        inference_mode=inference)
                            exp = ora_trainable(n_ln, hidden, base, adapters,
                                                active, inference)
                            assert got == exp, ("trainable %r" % ((n_ln, hidden,
                                base, adapters, active, inference),))
                            n += 1
    # 手算锚点：n=2 hidden=4 -> 16 params per adapter
    g = solve(mode="trainable", n_target_ln=2, hidden=4, base_params=100,
              adapters=["a", "b"], active_adapter="a", inference_mode=False)
    assert g["trainable_params"] == 16, g
    assert g["frozen_params"] == 100 + 32 - 16, g
    assert g["per_adapter"]["a"]["requires_grad"] is True
    assert g["per_adapter"]["b"]["requires_grad"] is False
    assert g["base_frozen"] is True
    g2 = solve(mode="trainable", n_target_ln=2, hidden=4, base_params=100,
               adapters=["a", "b"], active_adapter="a", inference_mode=True)
    assert g2["trainable_params"] == 0 and g2["frozen_params"] == 132, g2
    print("trainable: %d grid + 2 anchors OK" % n)


def check_forward(solve):
    n = 0
    for disable in (False, True):
        for merged in (False, True):
            for active in ([], ["a"], ["a", "b"]):
                for present in (False, True):
                    for base_out, tuned_out in (("BASE", "TUNED"), (0.25, 9.0)):
                        got = solve(mode="forward", disable_adapters=disable,
                                    merged=merged, active_adapters=active,
                                    adapter_present=present,
                                    base_out=base_out, tuned_out=tuned_out)
                        exp = ora_forward(disable, merged, active, present,
                                          base_out, tuned_out)
                        assert got == exp, ("forward %r" % ((disable, merged,
                            active, present),))
                        n += 1
    # 手算锚点：五分支优先级
    assert solve(mode="forward", disable_adapters=True, merged=True,
                 active_adapters=["a"], adapter_present=True,
                 base_out="B", tuned_out="T") == {
        "branch": "disable_adapters", "out": "B", "error": None,
        "unmerged_on_disable": True}
    assert solve(mode="forward", disable_adapters=False, merged=True,
                 active_adapters=["a"], adapter_present=True,
                 base_out="B", tuned_out="T")["branch"] == "base"
    assert solve(mode="forward", active_adapters=[], adapter_present=True,
                 base_out="B", tuned_out="T")["branch"] == "base"
    assert solve(mode="forward", active_adapters=["a", "b"],
                 adapter_present=True, base_out="B", tuned_out="T")["error"] == \
        "multiple active adapters"
    assert solve(mode="forward", active_adapters=["a"], adapter_present=False,
                 base_out="B", tuned_out="T")["branch"] == "adapter_missing"
    assert solve(mode="forward", active_adapters=["a"], adapter_present=True,
                 base_out="B", tuned_out="T")["out"] == "T"
    print("forward: %d grid + 6 anchors OK" % n)


def check_merge_unmerge(solve):
    n = 0
    for action in ("merge", "unmerge"):
        for names in (None, ["a"], ["a", "b"], ["z"]):
            for merged in ([], ["a"]):
                for present in (["a"], []):
                    for safe_merge in (False, True):
                        for bs, ad in (("base", "a"), ("x", "y")):
                            got = solve(mode="merge_unmerge", action=action,
                                        adapter_names=names,
                                        active_adapters=["a"],
                                        present=present,
                                        merged_adapters=merged,
                                        base_slot=bs, adapter_slot=ad,
                                        safe_merge=safe_merge)
                            exp = ora_merge(action, names, ["a"], present,
                                            merged, bs, ad)
                            assert got == exp, ("merge %r" % ((action, names,
                                merged, present, bs, ad),))
                            n += 1
    # 手算锚点：merge 是交换、一次只允许一个、已 merged 先 unmerge、safe_merge 无效
    g = solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
              active_adapters=["a"], present=["a"], merged_adapters=[],
              base_slot="base", adapter_slot="a")
    assert g["base_slot"] == "a" and g["adapter_slot"] == "base", g
    assert g["merged_adapters"] == ["a"] and g["warning"] is None
    assert g["error"] is None and g["safe_merge_ignored"] is True
    g2 = solve(mode="merge_unmerge", action="merge",
               adapter_names=["a", "b"], active_adapters=["a"],
               present=["a"], merged_adapters=[], base_slot="base",
               adapter_slot="a")
    assert g2["error"] == "LN tuning does not allow merging more than one adapter"
    assert g2["base_slot"] == "base" and g2["merged_adapters"] == []
    g3 = solve(mode="merge_unmerge", action="merge", adapter_names=["z"],
               active_adapters=["a"], present=["a"], merged_adapters=[],
               base_slot="base", adapter_slot="a")
    assert g3["merged_adapters"] == [] and g3["base_slot"] == "base"
    g4 = solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
               active_adapters=["a"], present=["a"], merged_adapters=["a"],
               base_slot="a", adapter_slot="base")
    assert g4["warning"] == "Already merged. Unmerging first."
    assert g4["merged_adapters"] == ["a"] and g4["base_slot"] == "a"
    g5 = solve(mode="merge_unmerge", action="unmerge", adapter_names=None,
               active_adapters=["a"], present=["a"], merged_adapters=["a"],
               base_slot="a", adapter_slot="base")
    assert g5["merged_adapters"] == [] and g5["base_slot"] == "base"
    g6 = solve(mode="merge_unmerge", action="unmerge", adapter_names=None,
               active_adapters=["a"], present=["a"], merged_adapters=[],
               base_slot="base", adapter_slot="a")
    assert g6["warning"] == "Already unmerged. Nothing to do."
    # safe_merge 两种取值必须完全等价
    a = solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
              active_adapters=["a"], present=["a"], merged_adapters=[],
              base_slot="base", adapter_slot="a", safe_merge=False)
    b = solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
              active_adapters=["a"], present=["a"], merged_adapters=[],
              base_slot="base", adapter_slot="a", safe_merge=True)
    assert a == b, "safe_merge must not affect LN tuning merge"
    print("merge_unmerge: %d grid + 7 anchors OK" % n)


def check_unload(solve):
    keys = ["model.layers.0.input_layernorm", "model.norm",
            "model.ln_tuning_layers.a.weight", "model.layers.1.ln_tuning_.x",
            "model.classifier", "ln_tuning_.b"]
    with_base = ["model.layers.0.input_layernorm", "model.norm",
                 "model.classifier", "model.ln_tuning_layers.a.weight"]
    n = 0
    for active in (["a"], ["a", "b"]):
        for consider in (None, ["a"], ["a", "b"]):
            for mts in ({}, {"a": True}, {"b": True}):
                for merge in (False, True):
                    for prefix in ("ln_tuning_", "ln_tuning_"):
                        got = solve(mode="unload", active_adapters=active,
                                    adapters_to_consider=consider,
                                    modules_to_save_by_adapter=mts,
                                    named_keys=keys, prefix=prefix,
                                    keys_with_base_layer=with_base,
                                    merge=merge)
                        exp = ora_unload(active, consider, mts, prefix,
                                         keys, with_base, merge)
                        assert got == exp, ("unload %r" % ((active, consider,
                            mts, merge),))
                        n += 1
    # 手算锚点：prefix 过滤、多 adapter + modules_to_save 报错、replaced / merge_calls
    g = solve(mode="unload", active_adapters=["a"],
              adapters_to_consider=None, modules_to_save_by_adapter={},
              named_keys=keys, prefix="ln_tuning_",
              keys_with_base_layer=with_base, merge=True)
    assert g["error"] is None
    assert g["key_list"] == ["model.layers.0.input_layernorm", "model.norm",
                            "model.classifier"], g["key_list"]
    assert g["replaced"] == ["model.layers.0.input_layernorm", "model.norm",
                            "model.classifier"], g["replaced"]
    assert g["merge_calls"] == 3
    g2 = solve(mode="unload", active_adapters=["a"],
               adapters_to_consider=None, modules_to_save_by_adapter={},
               named_keys=keys, prefix="ln_tuning_",
               keys_with_base_layer=with_base, merge=False)
    assert g2["merge_calls"] == 0 and g2["replaced"] == g["replaced"]
    g3 = solve(mode="unload", active_adapters=["a", "b"],
               adapters_to_consider=None, modules_to_save_by_adapter={"a": True},
               named_keys=keys, prefix="ln_tuning_",
               keys_with_base_layer=with_base, merge=True)
    assert g3["error"] == \
        "Cannot unload multiple adapters that specify modules_to_save."
    assert g3["replaced"] == [] and g3["merge_calls"] == 0
    # 单个 adapter 带 modules_to_save 不报错
    g4 = solve(mode="unload", active_adapters=["a"],
               adapters_to_consider=None, modules_to_save_by_adapter={"a": True},
               named_keys=keys, prefix="ln_tuning_",
               keys_with_base_layer=with_base, merge=True)
    assert g4["error"] is None
    print("unload: %d grid + 4 anchors OK" % n)


def check(solve):
    check_select(solve)
    check_default_targets(solve)
    check_update(solve)
    check_trainable(solve)
    check_forward(solve)
    check_merge_unmerge(solve)
    check_unload(solve)
    print("all checks passed")
