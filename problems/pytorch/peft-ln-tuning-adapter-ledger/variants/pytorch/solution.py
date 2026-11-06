"""LN-Tuning 适配器生命周期账（order 834）。

回源：huggingface/peft main 分支
  src/peft/tuners/ln_tuning/config.py    LNTuningConfig
  src/peft/tuners/ln_tuning/layer.py     LNTuningLayer（deepcopy / merge / forward）
  src/peft/tuners/ln_tuning/model.py     LNTuningModel（_create_and_replace / unload）
  src/peft/tuners/tuners_utils.py        check_target_module_exists / _mark_only_adapters_as_trainable
  src/peft/utils/other.py                match_target_against_key = re.fullmatch
  src/peft/utils/constants.py            TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING

判题约定：单入口 solve(mode, **kw)，7 个 mode：
  select          目标模块选择（exclude / modules_to_save / target 匹配）
  default_targets 架构默认目标表
  update          update_layer 的 deepcopy 整层 + set_adapter 冻结
  trainable       _mark_only_adapters_as_trainable 的可训练参数账
  forward         LNTuningLayer.forward 五分支路由
  merge_unmerge   merge / unmerge 的「交换」状态机
  unload          _unloading_checks + key_list 过滤 + 模块换回

不进判题域（避免与既有题重复 / 需真实模型）：真实 nn.Module 与权重、LN 数值前向
（base_out / tuned_out 作显式入参）、layers_to_transform / layers_pattern、bias 配置分支、
ModulesToSaveWrapper 内部、tqdm 进度条、dtype autocast（官方 _cast_adapter_dtype 为空）。
"""

import copy
import re

TARGETS_BY_MODEL = {
    "llama": ["input_layernorm", "post_attention_layernorm", "norm"],
    "bloom": ["input_layernorm", "post_attention_layernorm", "ln_f"],
    "t5": ["layer_norm", "final_layer_norm"],
    "gpt2": ["ln_1", "ln_2", "ln_f"],
    "llava": ["multi_modal_projector", "input_layernorm",
              "post_attention_layernorm", "norm", "embed_tokens", "lm_head"],
    "bart": ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"],
}
DEFAULT_PREFIX = "ln_tuning_"


def _excluded_by(exclude, key):
    """官方 check_target_module_exists 的 exclude_modules 段。"""
    if not exclude:
        return False
    if isinstance(exclude, str):
        return re.fullmatch(exclude, key) is not None
    return key in exclude or any(key.endswith("." + x) for x in exclude)


def _excluded_by_mts(modules_to_save, key):
    """官方 modules_to_save 段：re.match(rf\"(^|.*\\.){m}($|\\..*)\", key)。"""
    if not modules_to_save:
        return False
    return any(re.match(r"(^|.*\.)%s($|\..*)" % m, key)
               for m in modules_to_save)


def solve(mode: str, **kw):
    if mode == "select":
        key = kw["key"]
        target_modules = kw["target_modules"]
        exclude_modules = kw.get("exclude_modules")
        modules_to_save = kw.get("modules_to_save")
        if _excluded_by(exclude_modules, key):
            return {"matched": False, "excluded": True,
                    "reason": "exclude_modules"}
        if _excluded_by_mts(modules_to_save, key):
            return {"matched": False, "excluded": True,
                    "reason": "modules_to_save"}
        if isinstance(target_modules, str):
            matched = re.fullmatch(target_modules, key) is not None
        elif key in target_modules or any(
                key.endswith("." + t) for t in target_modules):
            matched = True
        else:
            matched = False
        return {"matched": matched, "excluded": False,
                "reason": "ok" if matched else "no_match"}

    if mode == "default_targets":
        model_type = kw["model_type"]
        if model_type not in TARGETS_BY_MODEL:
            raise ValueError("unknown model_type: %s" % model_type)
        return {"targets": list(TARGETS_BY_MODEL[model_type]), "known": True}

    if mode == "update":
        weight = list(kw["weight"])
        bias = list(kw["bias"])
        adapter_name = kw["adapter_name"]
        active_adapter = kw["active_adapter"]
        inference_mode = bool(kw.get("inference_mode", False))
        in_features = int(kw["in_features"])
        # 官方 update_layer：ln_tuning_layers[adapter] = deepcopy(base_layer)
        copy_w = copy.deepcopy(weight)
        copy_b = copy.deepcopy(bias)
        original = {"weight": weight, "bias": bias}
        copied = {"weight": copy_w, "bias": copy_b}
        independent = (original["weight"] is not copied["weight"] and
                       original["bias"] is not copied["bias"])
        requires_grad = (not inference_mode) and (adapter_name == active_adapter)
        return {"original": original, "copy": copied,
                "independent": independent, "requires_grad": requires_grad,
                "in_features": in_features, "out_features": in_features}

    if mode == "trainable":
        n_target_ln = int(kw["n_target_ln"])
        hidden = int(kw["hidden"])
        base_params = int(kw.get("base_params", 0))
        adapters = list(kw["adapters"])
        active_adapter = kw["active_adapter"]
        inference_mode = bool(kw.get("inference_mode", False))
        per_adapter_params = n_target_ln * 2 * hidden
        per_adapter = {}
        trainable = 0
        for name in adapters:
            rg = (name == active_adapter) and (not inference_mode)
            per_adapter[name] = {"requires_grad": rg,
                                 "params": per_adapter_params}
            if rg:
                trainable += per_adapter_params
        total_adapter_params = per_adapter_params * len(adapters)
        return {"trainable_params": trainable,
                "frozen_params": base_params + total_adapter_params - trainable,
                "per_adapter": per_adapter, "base_frozen": True}

    if mode == "forward":
        disable = bool(kw.get("disable_adapters", False))
        merged = bool(kw.get("merged", False))
        active = list(kw.get("active_adapters", []))
        present = bool(kw.get("adapter_present", True))
        base_out = kw["base_out"]
        tuned_out = kw["tuned_out"]
        error = None
        if disable:
            branch, out = "disable_adapters", base_out
            unmerged = merged
        elif merged or len(active) == 0:
            branch, out = "base", base_out
            unmerged = False
        elif len(active) != 1:
            branch, out = "error", None
            error = "multiple active adapters"
            unmerged = False
        elif not present:
            branch, out = "adapter_missing", base_out
            unmerged = False
        else:
            branch, out = "adapter", tuned_out
            unmerged = False
        return {"branch": branch, "out": out, "error": error,
                "unmerged_on_disable": unmerged}

    if mode == "merge_unmerge":
        action = kw["action"]
        active_adapters = list(kw.get("active_adapters", []))
        names = kw.get("adapter_names")
        names = list(names) if names is not None else list(active_adapters)
        present = set(kw.get("present", []))
        merged_adapters = list(kw.get("merged_adapters", []))
        base_slot = kw.get("base_slot", "base")
        adapter_slot = kw.get("adapter_slot", "base")
        warning = None
        error = None
        if action == "merge":
            if len(names) > 1:
                error = "LN tuning does not allow merging more than one adapter"
            elif not names or names[0] not in present:
                pass
            else:
                if merged_adapters:
                    warning = "Already merged. Unmerging first."
                    base_slot, adapter_slot = adapter_slot, base_slot
                    merged_adapters = []
                base_slot, adapter_slot = adapter_slot, base_slot
                merged_adapters = [names[0]]
        elif action == "unmerge":
            if not merged_adapters:
                warning = "Already unmerged. Nothing to do."
            else:
                merged_adapters = merged_adapters[:-1]
                base_slot, adapter_slot = adapter_slot, base_slot
        else:
            raise ValueError("unknown merge action")
        return {"base_slot": base_slot, "adapter_slot": adapter_slot,
                "merged_adapters": merged_adapters, "warning": warning,
                "error": error, "safe_merge_ignored": True}

    if mode == "unload":
        active = list(kw.get("active_adapters", []))
        consider = kw.get("adapters_to_consider")
        considered = list(consider) if consider is not None else list(active)
        mts = dict(kw.get("modules_to_save_by_adapter", {}))
        prefix = kw.get("prefix", DEFAULT_PREFIX)
        named_keys = list(kw["named_keys"])
        with_base = set(kw.get("keys_with_base_layer", []))
        merge = bool(kw.get("merge", True))
        error = None
        if any(mts.get(a) for a in considered) and len(considered) > 1:
            error = "Cannot unload multiple adapters that specify modules_to_save."
        key_list = [k for k in named_keys if prefix not in k]
        replaced = [k for k in key_list if k in with_base] if error is None else []
        merge_calls = len(replaced) if merge else 0
        return {"error": error, "key_list": key_list,
                "replaced": replaced, "merge_calls": merge_calls}

    raise ValueError("unknown mode: %s" % mode)
