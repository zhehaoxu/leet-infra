# 踩坑与回源记录（order 834）

## 回源口径

- 仓 `huggingface/peft`，`main` 分支；`utils/other.py::match_target_against_key` 通过 GitHub API `contents` + `base64.b64decode` 读取（raw 通道在 833 期已证实不稳），其余源文件下载到 `/tmp/peft_ln_src/`。

```
src/peft/tuners/ln_tuning/config.py    LNTuningConfig
src/peft/tuners/ln_tuning/layer.py     LNTuningLayer (deepcopy / merge / forward)
src/peft/tuners/ln_tuning/model.py     LNTuningModel (_create_and_replace / unload)
src/peft/tuners/tuners_utils.py        check_target_module_exists / _mark_only_adapters_as_trainable
src/peft/utils/other.py                match_target_against_key = re.fullmatch
src/peft/utils/constants.py            TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING
```

- `check_target_module_exists` 的 `exclude_modules` 命中返回 `_ExcludedModule()`（**falsy 对象而非 False**），本题只判「是否排除」故直接实现为布尔判断；`modules_to_save` 用 `re.match(r"(^|.*\.)m($|\..*)", key)`（不是 fullmatch）。
- 回源对拍脚本 `/tmp/verify834.py` 用 `ast.get_source_segment` 抽官方 `_ExcludedModule` / `check_target_module_exists` / `LNTuningLayer.merge` / `unmerge` / `forward` 原样 `exec`（stub `nn.Module` 为普通对象），与自写 solve 网格对拍：**select 720 + merge/unmerge 96 + forward 24 = 840 组全一致**。

## 机制要点

- **模块选择优先级**：`exclude_modules` > `modules_to_save` > `target_modules`；str 型 target / exclude 一律 `re.fullmatch`，list 型用 `key in list` 或 `key.endswith("." + t)`（后缀必须带点）。
- **update_layer 是整层 deepcopy**：`ln_tuning_layers[adapter] = deepcopy(base_layer)`，复制 weight + bias 两个独立对象，不是新建小适配器；`set_adapter` 让非 active adapter 的整层 `requires_grad_(False)`。判题用「副本对象与原件不是同一对象」断言（`is not`）暴露 deepcopy 语义，不能只判值相等。
- **merge/unmerge 是交换不是合并**：核心一行把 `base_layer` 与副本互换；`safe_merge` 参数完全无效；一次只允许 1 个 adapter（`>1` 抛错）；已 merged 先 warn 再 unmerge；`merged_adapters` 是 append/pop 栈。官方 `unmerge()` 交换用的 key 来自 `merged_adapters.pop()`，本题对拍时固定为 `"a"`。
- **forward 五分支**：`disable_adapters` → `merged or len(active)==0` → 多 active 报错 → 副本缺失 → 正常 adapter；`disable` 且已 merged 时官方会先 unmerge，故 `unmerged_on_disable=merged`。
- **`_unloading_checks`**：只有当「任一 considered adapter 带 modules_to_save」**且** `len(considered) > 1` 才报错；单个 adapter 带 modules_to_save 不报错。
- **`key_list` 过滤**：`[k for k in named_keys if prefix not in k]`，跳过含 `ln_tuning_` 前缀的 key；`replaced` 只取 `key_list` 中带 `base_layer` 的 key，`merge_calls = len(replaced) if merge else 0`，error 时整体置空。

## 测试设计

- 7 个 `check_*`：`select / default_targets / update / trainable / forward / merge_unmerge / unload`，judge 双模式 `run` 与 `submit` 均 `7/7 ACCEPTED`。
- 独立 oracle 用纯 Python 重写，不 import solution；每个 mode 全网格对拍 + 手算锚点。`select` 覆盖 str/list target、str/list exclude、modules_to_save 正则、后缀带点与否、`my_input_layernorm` 反例；`update` 网格覆盖 deepcopy 独立性与 inference/active 组合；`merge_unmerge` 覆盖 safe_merge 两种取值等价、已 merged 先 unmerge、多 adapter 报错；`unload` 覆盖 prefix 过滤、error 清空、merge 开关。
- 中间量必须暴露：`independent`、`per_adapter`、`merged_adapters` 栈、`base_slot/adapter_slot`、`warning/error`、`key_list/replaced/merge_calls`、`branch/unmerged_on_disable` 全部进返回值，否则分支顺序类与交换类变异不可观测。

## mutation 设计（39 变异全 CAUGHT）

- select 族：fullmatch→match（target 与 exclude 各一）、exclude 直接放行、modules_to_save 正则放宽、后缀去点、`key in list` 精确项被删、exclude 优先级反转；
- default 族：未知架构静默返回、llama 默认表改错；
- update 族：deepcopy→引用（weight / bias 各一）、inference 反转、active 反转、out_features 错；
- trainable 族：漏算 bias、inference 反转、active 反转、frozen 账错、base_frozen 写反；
- forward 族：disable 分支失效、无 active 分支丢失、多 active 阈值、present 反、out 取错、unmerged_on_disable 写反；
- merge 族：多 adapter 阈值、present 检查、已 merged 的 unmerge 交换、主交换、栈重复、safe_merge 生效、unmerge 不 pop / 不交换；
- unload 族：modules_to_save 少判 len>1、prefix 过滤取反、error 时 replaced 不清空、merge_calls 不受 merge 控制、considered 取空。
- **锚点踩坑**：`select-suffix-no-dot` 的目标表达式在源码里跨两行（`any(` 换行 + 16 空格缩进），单行锚点 `SRC.count==0` 被误报 SURVIVED；改成带换行的精确片段后 `39/39 CAUGHT, 0 SURVIVED`。这是脚本 anchor error，不是语义变异存活。
- 真等价剔除：无（39 条全部可观测）。

## 查重划界

- 与 `peft-bitfit-count`（只训练 bias，无整层复制 / 无交换式 merge）、`peft-lora-merge`（数值 ΔW 合并，本题是指针交换）、`peft-multi-adapter-batch`（官方 LN-Tuning 明确禁止多 active / 多 merge）、`peft-vera-shared` / `peft-adaptionprompt-gated-attention` / `peft-polytropon-skill-routing`（共享基 / 门控 / 路由）均不同。
- 与 `operators` 侧 LayerNorm 数值前向题不同：本题**不判 LN 数值计算**，`base_out` / `tuned_out` 作显式入参，只考路由与状态。
- 红线：不跑真实模型 / 不依赖 transformers 权重；正则用纯 Python `re`；参数计数用维度入参；deepcopy 语义用对象独立断言。
