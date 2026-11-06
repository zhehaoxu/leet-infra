# LN-Tuning 的层复制、交换式 merge 与训练态账

## 背景

`huggingface/peft` 的 LN-Tuning 走的是一条和其它 PEFT 方法很不一样的路线：它不新建小适配器，而是把目标 LayerNorm **整层 deepcopy 一份**作为可训练副本；`merge` 也不是把增量权重加回原权重，而是把 `base_layer` 与副本做**指针交换**；推理时通过分支路由决定用原始层还是副本层。

本题把这条生命周期抽成**无状态判题**：所有模型结构名、参数、状态标志、输出值都作显式入参；不跑真实 `nn.Module`、不加载 transformers 权重、不判 LayerNorm 的数值前向（`base_out` / `tuned_out` 直接给定，只判选了哪条分支）。实现单入口 `solve(mode, **kw)`，共 7 个 mode。

## 通用约定

所有 mode 返回 dict，字段语义见「模式与规则」；未列出的中间量不判。`re` 一律用 Python `re.fullmatch` / `re.match`，字符串大小写敏感。默认目标表与前缀常量如下（判题也会显式传入，此处仅作口径说明）：

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

## 模式与规则

### 1. select：目标模块选择

输入 `key, target_modules, exclude_modules=None, modules_to_save=None`。按优先级判定，返回 `{"matched", "excluded", "reason"}`：

```
exclude_modules hit   -> excluded, reason="exclude_modules"
modules_to_save hit   -> excluded, reason="modules_to_save"
target_modules: str   -> re.fullmatch(pattern, key)
target_modules: list  -> key in list  OR  key.endswith("." + t)
otherwise             -> matched=False, reason="no_match"
```

- `exclude_modules` 为 str 时用 `re.fullmatch(exclude, key)`；为 list 时命中 `key in exclude` 或 `key.endswith("." + x)`。命中即 `matched=False, excluded=True, reason="exclude_modules"`（优先级最高）。
- `modules_to_save` 为 list，任一 `m` 满足 `re.match(r"(^|.*\.)%s($|\..*)" % m, key)` 即排除，返回 `reason="modules_to_save"`。
- `target_modules` 为 str 时用 `re.fullmatch(target_modules, key)`（**不是 `re.match`**）；为 list 时 `key in list` 精确命中，或 `key.endswith("." + t)` 后缀命中（**必须带点分隔符**，`my_input_layernorm` 不命中 `input_layernorm`）。
- 命中返回 `matched=True, excluded=False, reason="ok"`；否则 `reason="no_match"`。

### 2. default_targets：架构默认目标表

输入 `model_type`。返回 `{"targets": [...], "known": True}`，`targets` 是 `TARGETS_BY_MODEL[model_type]` 的**副本**（顺序固定）。未知 `model_type` 抛 `ValueError`。

### 3. update：update_layer 的整层复制与冻结

输入 `weight, bias, adapter_name, active_adapter, inference_mode=False, in_features`。返回 `{"original", "copy", "independent", "requires_grad", "in_features", "out_features"}`。

- `original` = `{"weight": weight, "bias": bias}`；`copy` = 对两者各做 `deepcopy`（**副本是与原件不同的对象**，但值相等）；
- `independent` = 副本 weight 与原件 weight **不是同一对象** 且 副本 bias 与原件 bias 不是同一对象；
- `requires_grad` = `(not inference_mode) and (adapter_name == active_adapter)`（非 active adapter 新建的整层被冻结）；
- `out_features = in_features`（LN 层输入输出同维）。

### 4. trainable：可训练参数账

输入 `n_target_ln, hidden, base_params=0, adapters, active_adapter, inference_mode=False`。每个 adapter 的参数量为 `n_target_ln * 2 * hidden`（weight + bias）。返回 `{"trainable_params", "frozen_params", "per_adapter", "base_frozen"}`：

- `per_adapter[name] = {"requires_grad": (name == active_adapter) and (not inference_mode), "params": n_target_ln * 2 * hidden}`；
- `trainable_params` = 所有 `requires_grad` 为真的 adapter 参数量之和；
- `frozen_params` = `base_params + (n_target_ln * 2 * hidden * len(adapters)) - trainable_params`；
- `base_frozen` 恒为 `True`（基础模型全部冻结）。

### 5. forward：五分支路由

输入 `disable_adapters=False, merged=False, active_adapters=[], adapter_present=True, base_out, tuned_out`。按优先级返回 `{"branch", "out", "error", "unmerged_on_disable"}`：

1. `disable_adapters` → `branch="disable_adapters"`, `out=base_out`, `unmerged_on_disable=merged`（若已 merged，官方会先 unmerge 再用 base_layer）；
2. `merged` 或 `len(active_adapters) == 0` → `branch="base"`, `out=base_out`；
3. `len(active_adapters) != 1` → `branch="error"`, `out=None`, `error="multiple active adapters"`（LN-Tuning 不支持多 adapter 同时推理）；
4. `not adapter_present` → `branch="adapter_missing"`, `out=base_out`；
5. 否则 → `branch="adapter"`, `out=tuned_out`。

`error` 仅第 3 分支非 `None`，`unmerged_on_disable` 仅第 1 分支可能为 `True`，其余分支为 `False`。

### 6. merge_unmerge：交换式状态机

输入 `action, adapter_names=None, active_adapters=[], present=[], merged_adapters=[], base_slot="base", adapter_slot="base", safe_merge=False`。返回 `{"base_slot", "adapter_slot", "merged_adapters", "warning", "error", "safe_merge_ignored"}`。

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

- `names` = `adapter_names` 若非 `None`，否则取 `active_adapters`；
- `merge` 的核心是**交换** `base_slot` 与 `adapter_slot`（对应官方 `self.base_layer, self.ln_tuning_layers[name] = self.ln_tuning_layers[name], self.base_layer`），不是数值合并；`safe_merge` 参数**完全无效**，`safe_merge_ignored` 恒为 `True`；
- 已 `merged_adapters` 非空时先 `warning="Already merged. Unmerging first."` 并交换回一次，再执行本次 merge 的交换；
- `unmerge` 弹出一个 `merged_adapters` 元素后交换一次；栈为空时只给 warning；
- `merged_adapters` 是列表（栈），用 `append/pop` 语义。

### 7. unload：卸载检查与 key 过滤

输入 `active_adapters=[], adapters_to_consider=None, modules_to_save_by_adapter={}, named_keys, prefix="ln_tuning_", keys_with_base_layer=[], merge=True`。返回 `{"error", "key_list", "replaced", "merge_calls"}`。

- `considered` = `adapters_to_consider` 若非 `None`，否则取 `active_adapters`；
- 若 `any(modules_to_save_by_adapter.get(a) for a in considered)` **且** `len(considered) > 1` → `error="Cannot unload multiple adapters that specify modules_to_save."`（`_unloading_checks`）；
- `key_list = [k for k in named_keys if prefix not in k]`（**跳过含 `ln_tuning_` 前缀的 key**）；
- `error` 非 `None` 时 `replaced=[]`、`merge_calls=0`；否则 `replaced = [k for k in key_list if k in keys_with_base_layer]`，`merge_calls = len(replaced) if merge else 0`。

## 输入与返回示例

```
solve(mode="select", key="model.layers.0.input_layernorm",
      target_modules=["input_layernorm"])
# {"matched": True, "excluded": False, "reason": "ok"}

solve(mode="merge_unmerge", action="merge", adapter_names=["a"],
      present=["a"], merged_adapters=[], base_slot="base", adapter_slot="a")
# {"base_slot": "a", "adapter_slot": "base", "merged_adapters": ["a"],
#  "warning": None, "error": None, "safe_merge_ignored": True}
```

## 不进判题域

真实 `nn.Module` 与权重、LayerNorm 数值前向、layers_to_transform / layers_pattern、`bias` 配置分支、ModulesToSaveWrapper 内部、`tqdm` 进度条、dtype autocast（官方 `_cast_adapter_dtype` 为空）、以及任何 transformers 权重加载。判题只考上述控制面的确定性账。
