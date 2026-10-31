# Notes: nn.Module Registration Ledger

## Provenance

PyTorch docs *Module* plus the `__setattr__` / `named_parameters` / `state_dict` implementations in `torch/nn/modules/module.py`, verified on torch 2.14.1:

- **Traversal order**: `named_parameters` / `named_buffers` / `state_dict` all yield "own `_parameters` → own `_buffers` → child `_modules`" — parameters of an `nn.Linear` assigned **first** in `__init__` come **after** a bare `Parameter` attribute assigned later. Intuition says assignment order; reality is bucketing by container class, each bucket ordered.
- **Shared-parameter dedup**: assigning one `Parameter` object to two attributes — `named_parameters()` (default `remove_duplicate=True`) yields only the **first-registered name**; `remove_duplicate=False` yields both. **`state_dict()` does not dedup** — both keys appear, and `load_state_dict` will write the same parameter twice (harmless value-wise, but the key ledger has an extra entry).
- **Buffer persistence**: a `persistent=False` buffer shows up in `named_buffers()` but not in `state_dict()`; re-assigning an ordinary `Tensor` to a registered buffer attribute **keeps the registration** (`__setattr__` sees the name already in `_buffers` and swaps the value), and it stays in `state_dict`. `.to(dtype)` rewrites buffers in place.
- **Registration blind spots**: a plain Python `list` of modules is **not registered** (`modules()` never sees it, its parameters are invisible) — the reason `ModuleList`/`ModuleDict` exist; `register_parameter(name, None)` is legal (a placeholder) but the name is **invisible** in both `named_parameters` and `state_dict`.
- `__setattr__` auto-registers `Parameter` objects into `_parameters`; an ordinary `Tensor` on an unregistered name is just an attribute.

## Key points and traps

1. **The `module_count=4` ledger**: root + `lin1` + `blocks` + `blocks.0`. The Linear inside `loose` is absent (`loose_in_modules=False`), so `module_linear_count=2`.
2. **`state_dict` key order**: `["p_first", "p_second", "running", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]` — own parameters (including the duplicate key) → persistent buffers → submodules; `scratch` (non-persistent) and `placeholder` (None) are both absent.
3. **`nodup - dedup = 1`**: the single shared pair contributes exactly one difference, invariant across random sizes.
4. **Two experiment fields**: `buffer_reassign_kept` (registration survives a re-assign of an ordinary Tensor) and `buffer_follows_to` (buffer dtype follows `.to(float64)`) are both True, matching the real `__setattr__` and `_apply` implementations.
5. **bool is an int subclass**: `solve(True, 1, 1)` must be rejected — the explicit `isinstance(v, bool)` check cannot be dropped.

## Mutation testing (9/9 caught)

dedup ledger swapped to non-dedup, `dedup_keeps_first` inverted, `scratch` made persistent, `placeholder` given a real parameter, `loose` turned into a `ModuleList`, both experiment fields hardcoded False, `.to()` removed, bool validation dropped, `p_second` replaced by an independent object. All caught by anchor/blind-spot/random cases; no equivalent mutation.

## Complexity

An O(#parameters) registration ledger; microseconds at widths ≤ 24.
