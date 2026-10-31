# nn.Module Registration Ledger: Parameter Dedup, Buffer Persistence and Registration Blind Spots

## Problem

`nn.Module` parameter collection, buffer management and `state_dict` all rest on the registration mechanism, which hides five counter-intuitive behaviors: traversal buckets by container class, shared parameters are only half-deduplicated, non-persistent buffers vanish from `state_dict`, plain lists are a registration blind spot, and None parameters are legal placeholders. This problem asks you to read the complete ledger of a module that mixes every registration style.

## Rules

Implement `solve(in_dim, hidden, out_dim)` (three positive integer widths). Build the module per the starter comments and return the ledger dict:

| Field | Anchor (4, 6, 3) | Tested concept |
|---|---|---|
| `param_names_dedup` | `["p_first", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]` | own parameters precede submodules; shared param keeps the first name |
| `param_names_nodup` | the above plus `"p_second"`, 6 total | `remove_duplicate=False` differs by exactly 1 |
| `buffer_names` | `["running", "scratch"]` | named_buffers ignores persistence |
| `state_dict_keys` | `["p_first", "p_second", "running", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]` | **no dedup**; non-persistent and None both absent |
| `sd_has_duplicate_key` | `True` | state_dict emits two keys for a shared parameter |
| `placeholder_in_named` / `placeholder_in_sd` | `False` / `False` | None parameters are invisible |
| `loose_in_modules` | `False` | plain lists are not registered |
| `module_count` / `module_linear_count` | `4` / `2` | root + lin1 + blocks + blocks.0 |
| `buffer_reassign_kept` / `buffer_follows_to` | `True` / `True` | re-assigning an ordinary Tensor keeps registration; buffers follow `.to()` |

Requirements:

1. Raise `ValueError` when any width is not a positive integer (bools, floats, strings, 0 and negatives included).
2. Ledger fields must come from a real module — no hand-filled constants.

## Example

```
Input: in_dim=4, hidden=6, out_dim=3
5 deduped parameters, 6 un-deduped; 4 modules (the list-wrapped Linear doesn't count)
7 state_dict keys (p_second present; scratch and placeholder absent)
```

## Constraints

- Widths within 1–8 suffice; the ledger's ordering (parameters before submodules, key order) is invariant across sizes
- Random-size cases re-verify `nodup - dedup == 1` and the first three keys' order
