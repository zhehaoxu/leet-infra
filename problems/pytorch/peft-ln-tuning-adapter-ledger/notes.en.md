# Pitfalls and source-verification notes (order 834)

## Source verification

- Repo `huggingface/peft`, `main` branch; `utils/other.py::match_target_against_key` was read via the GitHub API `contents` endpoint plus `base64.b64decode` (the raw channel proved unstable in the 833 round), the other sources were downloaded to `/tmp/peft_ln_src/`.

```
src/peft/tuners/ln_tuning/config.py    LNTuningConfig
src/peft/tuners/ln_tuning/layer.py     LNTuningLayer (deepcopy / merge / forward)
src/peft/tuners/ln_tuning/model.py     LNTuningModel (_create_and_replace / unload)
src/peft/tuners/tuners_utils.py        check_target_module_exists / _mark_only_adapters_as_trainable
src/peft/utils/other.py                match_target_against_key = re.fullmatch
src/peft/utils/constants.py            TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING
```

- A hit on `exclude_modules` in `check_target_module_exists` returns `_ExcludedModule()` (a **falsy object, not False**); this problem only judges whether the key is excluded, so it is implemented directly as a boolean check. `modules_to_save` uses `re.match(r"(^|.*\.)m($|\..*)", key)` (not fullmatch).
- The comparison script `/tmp/verify834.py` extracts the official `_ExcludedModule` / `check_target_module_exists` / `LNTuningLayer.merge` / `unmerge` / `forward` with `ast.get_source_segment` and `exec`s them unchanged (stubbing `nn.Module` as a plain object), comparing against the self-written solve over a grid: **select 720 + merge/unmerge 96 + forward 24 = 840 cases, all identical**.

## Mechanism details

- **Module-selection priority**: `exclude_modules` > `modules_to_save` > `target_modules`; str-typed target / exclude always use `re.fullmatch`, list-typed use `key in list` or `key.endswith("." + t)` (the suffix requires a dot).
- **update_layer deep-copies the whole layer**: `ln_tuning_layers[adapter] = deepcopy(base_layer)`, copying weight + bias as two independent objects, not building a small adapter; `set_adapter` freezes the entire layer of a non-active adapter with `requires_grad_(False)`. Judging exposes the deepcopy semantics with an "the copy is not the same object as the original" assertion (`is not`), not merely value equality.
- **merge/unmerge is a swap, not a merge**: the core line exchanges `base_layer` and the replica; the `safe_merge` argument has no effect at all; only 1 adapter is allowed at a time (`>1` raises); an already-merged state warns then unmerges first; `merged_adapters` is an append/pop stack. The official `unmerge()` swaps using the key from `merged_adapters.pop()`, fixed to `"a"` in this comparison.
- **forward has five branches**: `disable_adapters` -> `merged or len(active)==0` -> multiple active error -> replica missing -> normal adapter; when `disable` and already merged, the official code unmerges first, hence `unmerged_on_disable=merged`.
- **`_unloading_checks`**: raises only when "any considered adapter carries modules_to_save" **and** `len(considered) > 1`; a single adapter with modules_to_save does not raise.
- **`key_list` filtering**: `[k for k in named_keys if prefix not in k]`, skipping keys containing the `ln_tuning_` prefix; `replaced` takes only keys in `key_list` that have a `base_layer`, `merge_calls = len(replaced) if merge else 0`, and everything is cleared when an error is set.

## Test design

- 7 `check_*` functions: `select / default_targets / update / trainable / forward / merge_unmerge / unload`; judge reports `7/7 ACCEPTED` in both `run` and `submit` modes.
- An independent oracle is rewritten in pure Python with no import of solution; every mode uses a full-grid comparison plus hand-computed anchors. `select` covers str/list targets, str/list excludes, the modules_to_save regex, dotted vs undotted suffixes, and the `my_input_layernorm` counter-example; `update` grids over deepcopy independence and inference/active combinations; `merge_unmerge` covers the equivalence of both safe_merge values, unmerge-first on an already-merged state, and the multi-adapter error; `unload` covers prefix filtering, clearing on error, and the merge toggle.
- Intermediate values must be exposed: `independent`, `per_adapter`, the `merged_adapters` stack, `base_slot/adapter_slot`, `warning/error`, `key_list/replaced/merge_calls`, and `branch/unmerged_on_disable` all appear in the return value; otherwise branch-order and swap mutants are invisible.

## Mutation design (39 mutants, all CAUGHT)

- Select family: fullmatch -> match (one each for target and exclude), letting exclude through, loosening the modules_to_save regex, dropping the dot from the suffix, deleting the exact `key in list` branch, inverting exclude priority;
- Default family: silently returning on an unknown architecture, corrupting the llama default table;
- Update family: deepcopy -> reference (weight / bias each), inverted inference, inverted active, wrong out_features;
- Trainable family: dropping bias from the count, inverted inference, inverted active, wrong frozen ledger, flipped base_frozen;
- Forward family: disabled disable branch, lost no-active branch, multi-active threshold, inverted present, wrong out, flipped unmerged_on_disable;
- Merge family: multi-adapter threshold, present check, unmerge-swap on an already-merged state, main swap, duplicated stack, safe_merge taking effect, unmerge without pop / without swap;
- Unload family: modules_to_save missing the len>1 check, inverted prefix filter, replaced not cleared on error, merge_calls ignoring merge, empty considered.
- **Anchor trap**: the target expression of `select-suffix-no-dot` spans two lines in the source (`any(` plus a newline and 16-space indent), so the single-line anchor had `SRC.count==0` and was wrongly reported as SURVIVED; switching to an exact fragment with the newline gave `39/39 CAUGHT, 0 SURVIVED`. This was a script anchor error, not a semantic survivor.
- Truly equivalent mutants removed: none (all 39 are observable).

## Duplicate-avoidance boundaries

- Distinct from `peft-bitfit-count` (bias-only training, no whole-layer copy / no swap-style merge), `peft-lora-merge` (numeric delta-W merge, while this is a pointer swap), `peft-multi-adapter-batch` (official LN-Tuning explicitly forbids multiple active / merged adapters), `peft-vera-shared` / `peft-adaptionprompt-gated-attention` / `peft-polytropon-skill-routing` (shared basis / gating / routing).
- Distinct from the `operators`-side numeric LayerNorm forward problems: this problem **does not judge LN numeric computation**; `base_out` / `tuned_out` are explicit inputs and only routing and state are judged.
- Red lines: no real model / no transformers weights; regex uses pure Python `re`; parameter counts use dimension inputs; deepcopy semantics use an object-independence assertion.
