# GenKnowSub Implementation Notes

## Source and Structure

- Paper arXiv:2505.10939 (ACL 2025 short, University of Tehran), official repo `saharsamr/Modular-LLM`; the PEFT-side implementation lives in `src/peft/tuners/lora/arrow.py::gen_know_sub` (landed with the same PR #2644 as Arrow, via `use_gks`/`gks_adapter_names`/`gks_done`/`gks_added_adapter_names`).
- `gen_know_sub`: `avg_A = stack([lora_A[n].weight for n in gks_adapter_names]).mean(0)`; `lora_A[name].weight.data.sub_(avg_A)` (same for B) rewrites the task factors IN PLACE. `gks_done=False` purifies everything; later runs only `gks_added_adapter_names`.
- Gate: `if not self.use_gks or self.gks_done and not self.gks_added_adapter_names: return` — Python precedence gives `not use_gks or (gks_done and not added)`.
- `create_arrow_model`: `use_gks=True` with empty `general_adapter_paths` raises ValueError; the `gks_`-prefixed adapters go through `check_loaded_lora_compatibility_arrow` together with the task adapters (shared rank/shapes/module set).
- Theory: "forgetting-via-negation" task arithmetic (arXiv:2212.04089).

## Judging Conventions and Quirks

- The subtraction acts on the **factors**: \((B-\bar B)(A-\bar A) = B@A - B@\bar A - \bar B@A + \bar B@\bar A\) — the cross-term identity is the strongest signal of this problem (m-sub-*/m-td-* mutations).
- The official in-place `data.sub_` becomes clone+sub in the judging domain (pure function); the bit-identical-input assertion guards against in-place pollution.
- For the incremental `only` subset: fancy-index assignment `a_new[idx] = a_new[idx].sub_(avg_a)` aligns both sides in the same order, so `sorted(only)` and `list(only)` are **genuinely equivalent** (not a usable mutation); non-selected experts are asserted with `torch.equal`.
- The gate is a three-state machine, not a boolean: `or`->`and`, `and`->`or`, `"all"`->`"added"`, `"added"`->`"skip"` each expose a different branch.
- Multiple language GK LoRAs (en/fr/de) entering the average is the paper's selling point — the mean mutations (`mean(0)`/`dim=0`/`mean(1)`) cover it.
- The pure mean-subtract account is small; the cross-term identity and the gate state machine complete the mutation surface.
