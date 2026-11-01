# weights_only Allowlist & state_dict Key Ledger

## Problem

`torch.load` defaults `weights_only=True` since 2.6: any GLOBAL instruction (custom class/function) in the pickle stream outside the allowlist raises `UnpicklingError`. Hand-write the decision layer — extract `archive/data.pkl` from the torch.save zip, scan GLOBAL instructions with `pickletools` (protocol 2 uses two-line text GLOBALs, no STACK_GLOBAL), and classify each scenario ok/rejected against a pinned allowlist table, plus a hand-written `load_state_dict` missing/unexpected key ledger. No `torch.load` / `load_state_dict` calls.

## Solution

Official semantics (`torch/serialization.py` + `torch/_weights_only_unpickler.py` v2.14.1):

- Pure primitive containers (dict/list/tuple/int/float/str/None/bool) emit no GLOBAL → naturally allowed;
- Allowlist (protocol-2 key form, reverse-engineered by measurement): `collections OrderedDict/Counter`, `torch Size/device/<dtype>`, `torch <dtype>Storage`, `torch._utils _rebuild_tensor_v2/_rebuild_parameter`, `__builtin__ set/bytearray/complex`, `_codecs encode`;
- Typical rejections: `torch.nn.modules.linear Linear` (any nn.Module), `types SimpleNamespace`, **`__builtin__ xrange` (range objects)**, numpy family (`numpy dtype/ndarray`, `numpy._core.multiarray _reconstruct`);
- lambda fails at **save time** with `PicklingError` (save_fail, not a load rejection);
- `load_state_dict`: missing = keys the model has but the state_dict lacks, unexpected the reverse, both reported **sorted**; `strict=True` with either non-empty → `RuntimeError` (first line `Error(s) in loading state_dict for ...`); `strict=False` → `IncompatibleKeys`; full match → `<All keys matched successfully>`;
- A bare `nn.Linear`'s state_dict keys are `weight/bias` — no attribute-name prefix (`fc.` only appears under container modules).

## Constraints

- 17 scenario objects (tensor dict, nested primitives, OrderedDict/Counter/Size/device/dtype/set/bytearray/complex/Parameter allowed; Linear/SimpleNamespace/range/numpy rejected; lambda save_fail).
- Three state_dict ledgers: full match, foreign model (`0.*` vs `weight/bias`), and a redundant `extra.weight` key.
