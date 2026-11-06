# LN-Tuning: layer copy, swap-style merge, and trainable-state ledger

## Background

LN-Tuning in `huggingface/peft` takes a route quite different from other PEFT methods: instead of building a small adapter, it **deep-copies the whole target LayerNorm** as the trainable replica; `merge` does not add a delta back into the weights, it **swaps the pointers** of `base_layer` and the replica; at inference a branch router decides whether to use the original layer or the replica layer.

This problem distills that lifecycle into **stateless judging**: all module names, parameters, state flags, and output values are explicit inputs; no real `nn.Module`, no transformers weights, no numeric LayerNorm forward (`base_out` / `tuned_out` are given directly, only the chosen branch is judged). Implement a single entry point `solve(mode, **kw)` with 7 modes.

## General conventions

Every mode returns a dict; field semantics are in "Modes and rules", and unlisted intermediate values are not judged. `re` always means Python `re.fullmatch` / `re.match`, case-sensitive. The default target table and prefix constant are below (judging also passes them explicitly; this is only for reference):

```
TARGETS_BY_MODEL = {
  "llama": ["input_layernorm", "post_attention_layernorm", "norm"],
  "bloom": ["input_layernorm", "post_attention_layernorm", "ln_f"],
  "t5":    ["layer_norm", "final_layer_norm"],
  "gpt2":  ["ln_1", "ln_2", "ln_f"],
  "llava": ["multi_modal_projector", "input_layernorm",
            "post_attention_layernorm", "norm", "embed_tokens", "lm_head"],
  "bart":  ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"],
}
DEFAULT_PREFIX = "ln_tuning_"
```

## Modes and rules

### 1. select: target-module selection

Inputs: `key, target_modules, exclude_modules=None, modules_to_save=None`. Decide in priority order; return `{"matched", "excluded", "reason"}`:

```
exclude_modules hit   -> excluded, reason="exclude_modules"
modules_to_save hit   -> excluded, reason="modules_to_save"
target_modules: str   -> re.fullmatch(pattern, key)
target_modules: list  -> key in list  OR  key.endswith("." + t)
otherwise             -> matched=False, reason="no_match"
```

- `exclude_modules` as str uses `re.fullmatch(exclude, key)`; as list it hits `key in exclude` or `key.endswith("." + x)`. A hit gives `matched=False, excluded=True, reason="exclude_modules"` (highest priority).
- `modules_to_save` is a list; any `m` matching `re.match(r"(^|.*\.)%s($|\..*)" % m, key)` excludes the key with `reason="modules_to_save"`.
- `target_modules` as str uses `re.fullmatch(target_modules, key)` (**not `re.match`**); as list it matches `key in list` exactly, or `key.endswith("." + t)` as a suffix (**the dot separator is required**, so `my_input_layernorm` does not match `input_layernorm`).
- A hit returns `matched=True, excluded=False, reason="ok"`; otherwise `reason="no_match"`.

### 2. default_targets: architecture default target table

Inputs: `model_type`. Returns `{"targets": [...], "known": True}`, where `targets` is a **copy** of `TARGETS_BY_MODEL[model_type]` (fixed order). An unknown `model_type` raises `ValueError`.

### 3. update: whole-layer copy and freezing in update_layer

Inputs: `weight, bias, adapter_name, active_adapter, inference_mode=False, in_features`. Returns `{"original", "copy", "independent", "requires_grad", "in_features", "out_features"}`.

- `original` = `{"weight": weight, "bias": bias}`; `copy` = a `deepcopy` of each (**distinct objects**, equal values);
- `independent` = the copied weight is **not the same object** as the original weight AND the copied bias is not the same object as the original bias;
- `requires_grad` = `(not inference_mode) and (adapter_name == active_adapter)` (a newly built layer for a non-active adapter is frozen);
- `out_features = in_features` (a LayerNorm has equal input and output dimensions).

### 4. trainable: trainable-parameter ledger

Inputs: `n_target_ln, hidden, base_params=0, adapters, active_adapter, inference_mode=False`. Each adapter has `n_target_ln * 2 * hidden` parameters (weight + bias). Returns `{"trainable_params", "frozen_params", "per_adapter", "base_frozen"}`:

- `per_adapter[name] = {"requires_grad": (name == active_adapter) and (not inference_mode), "params": n_target_ln * 2 * hidden}`;
- `trainable_params` = sum over adapters whose `requires_grad` is true;
- `frozen_params` = `base_params + (n_target_ln * 2 * hidden * len(adapters)) - trainable_params`;
- `base_frozen` is always `True` (the base model is fully frozen).

### 5. forward: five-branch router

Inputs: `disable_adapters=False, merged=False, active_adapters=[], adapter_present=True, base_out, tuned_out`. Return `{"branch", "out", "error", "unmerged_on_disable"}` by priority:

1. `disable_adapters` → `branch="disable_adapters"`, `out=base_out`, `unmerged_on_disable=merged` (if already merged, the official code unmerges first, then uses base_layer);
2. `merged` or `len(active_adapters) == 0` → `branch="base"`, `out=base_out`;
3. `len(active_adapters) != 1` → `branch="error"`, `out=None`, `error="multiple active adapters"` (LN-Tuning does not support several active adapters at inference);
4. `not adapter_present` → `branch="adapter_missing"`, `out=base_out`;
5. otherwise → `branch="adapter"`, `out=tuned_out`.

`error` is non-`None` only in branch 3; `unmerged_on_disable` may be `True` only in branch 1, and is `False` in every other branch.

### 6. merge_unmerge: swap-style state machine

Inputs: `action, adapter_names=None, active_adapters=[], present=[], merged_adapters=[], base_slot="base", adapter_slot="base", safe_merge=False`. Returns `{"base_slot", "adapter_slot", "merged_adapters", "warning", "error", "safe_merge_ignored"}`.

```
merge:
  len(names) > 1            -> error "LN tuning does not allow merging more than one adapter"
  names[0] not in present    -> no-op
  merged_adapters non-empty  -> warning "Already merged. Unmerging first."; swap back
  swap base_slot <-> adapter_slot; merged_adapters = [names[0]]
unmerge:
  merged_adapters empty      -> warning "Already unmerged. Nothing to do."
  else                       -> pop one; swap base_slot <-> adapter_slot
```

- `names` = `adapter_names` if not `None`, else `active_adapters`;
- the core of `merge` is a **swap** of `base_slot` and `adapter_slot` (the official `self.base_layer, self.ln_tuning_layers[name] = self.ln_tuning_layers[name], self.base_layer`), not a numeric merge; the `safe_merge` argument has **no effect at all**, and `safe_merge_ignored` is always `True`;
- when `merged_adapters` is non-empty, first emit `warning="Already merged. Unmerging first."` and swap back once, then perform this merge's swap;
- `unmerge` pops one `merged_adapters` element and swaps once; when the stack is empty it only emits a warning;
- `merged_adapters` is a list (stack) with `append/pop` semantics.

### 7. unload: unload checks and key filtering

Inputs: `active_adapters=[], adapters_to_consider=None, modules_to_save_by_adapter={}, named_keys, prefix="ln_tuning_", keys_with_base_layer=[], merge=True`. Returns `{"error", "key_list", "replaced", "merge_calls"}`.

- `considered` = `adapters_to_consider` if not `None`, else `active_adapters`;
- if `any(modules_to_save_by_adapter.get(a) for a in considered)` **and** `len(considered) > 1` → `error="Cannot unload multiple adapters that specify modules_to_save."` (`_unloading_checks`);
- `key_list = [k for k in named_keys if prefix not in k]` (**keys containing the `ln_tuning_` prefix are skipped**);
- when `error` is non-`None`, `replaced=[]` and `merge_calls=0`; otherwise `replaced = [k for k in key_list if k in keys_with_base_layer]` and `merge_calls = len(replaced) if merge else 0`.

## Input and output examples

```
solve(mode="select", key="model.layers.0.input_layernorm",
      target_modules=["input_layernorm"])
# {"matched": True, "excluded": False, "reason": "ok"}

solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
      present=["a"], merged_adapters=[], base_slot="base", adapter_slot="a")
# {"base_slot": "a", "adapter_slot": "base", "merged_adapters": ["a"],
#  "warning": None, "error": None, "safe_merge_ignored": True}
```

## Out of scope

Real `nn.Module` and weights, numeric LayerNorm forward, layers_to_transform / layers_pattern, `bias` config branches, ModulesToSaveWrapper internals, `tqdm` progress bars, dtype autocast (the official `_cast_adapter_dtype` is empty), and any transformers weight loading. Judging covers only the deterministic ledger of the control plane above.
