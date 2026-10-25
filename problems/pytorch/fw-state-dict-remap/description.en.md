# Checkpoint Key Remapping and strict=False Accounting

## Problem

After a model redesign, the old checkpoint's key names no longer match the new modules: prefixes are renamed, `w/b/ln` become `weight/bias/layernorm`, the Conv1D-style `embed.w` must be transposed into the Linear-style `embed.weight`, the old language-model head is dropped, and the new model adds `gate` layers the checkpoint does not have. This is the daily grind of HF `convert_*_checkpoint` scripts. This problem asks you to complete one full pass of "remap → load with `strict=False` → keep all four ledgers".

## Rules

Implement `solve()` returning a `dict` with four ledgers: `mapping` (rename pairs), `missing_keys`, `unexpected_keys`, `loaded` (parameter values after loading).

The new model structure, the old checkpoint's keys/shapes/deterministic values, the remapping rules and the return format are all given in the starter docstring (the single source of truth). Key points:

1. Checkpoint values are consecutive integer segments (\(\texttt{arange} + \texttt{cursor}\)); no randomness is involved.
2. `transformer.h.{i}.*` is renamed to `blocks.{i}.*`; `embed.w` becomes `embed.weight` **with the value transposed** (old (3,2) → new (2,3), flattened row-major).
3. Load with `model.load_state_dict(remapped, strict=False)`; you **must sort** `missing_keys`/`unexpected_keys` yourself before returning (the framework does not guarantee order).
4. `loaded` takes the post-load `model.state_dict()`, each parameter flattened row-major into a `list[float]`; parameters not loaded keep their zero initialization.

## Examples

```
Input: none
Output:
{
 "mapping": {
  "embed.w": "embed.weight",
  "transformer.h.0.b": "blocks.0.bias",
  "transformer.h.0.ln": "blocks.0.layernorm",
  "transformer.h.0.w": "blocks.0.weight",
  "transformer.h.1.b": "blocks.1.bias",
  "transformer.h.1.ln": "blocks.1.layernorm",
  "transformer.h.1.w": "blocks.1.weight"
 },
 "missing_keys": ["blocks.0.gate", "blocks.1.gate"],
 "unexpected_keys": ["lm_head.weight"],
 "loaded": {
  "embed.weight":     [25.0, 27.0, 29.0, 26.0, 28.0, 30.0],
  "blocks.0.weight":  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
  "blocks.0.bias":    [6.0, 7.0],
  "blocks.0.layernorm": [8.0, 9.0, 10.0],
  "blocks.0.gate":    [0.0, 0.0],
  "blocks.1.weight":  [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
  "blocks.1.bias":    [19.0, 20.0, 21.0, 22.0],
  "blocks.1.layernorm": [23.0, 24.0],
  "blocks.1.gate":    [0.0, 0.0, 0.0, 0.0]
 }
}
```

Note the transposition effect on `embed.weight`: the old values \(\begin{pmatrix}25&26\\27&28\\29&30\end{pmatrix}\) flatten row-major after transposing as \(25,27,29,26,28,30\), not \(25,26,\dots,30\).

## Constraints

- Return a `dict` with exactly the keys `mapping` / `missing_keys` / `unexpected_keys` / `loaded`.
- All values are compared exactly (constructed from small integers, lossless in `float32`); the two key lists must be sorted `list[str]`.
- The flow is fully deterministic with no randomness; do not touch the global seed.
