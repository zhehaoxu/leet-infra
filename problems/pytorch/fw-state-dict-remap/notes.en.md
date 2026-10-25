# Explained: Checkpoint Key Remapping and strict=False Accounting

## Origin

`torch.nn.Module.load_state_dict` (official PyTorch docs): with `strict=False` it returns the named tuple `_IncompatibleKeys(missing_keys, unexpected_keys)`; in the source the two lists come from `set` differences then `sorted`. Note that a bare `nn.Parameter` attached to the root module appears in `state_dict` under the attribute name itself (e.g. `embed`, not `embed.weight`) — which is why this problem wraps it in a submodule. HF's `convert_gpt2_checkpoint.py` / `convert_llama_weights_to_hf.py` are the industrialized versions of exactly this "rename prefixes + transpose Conv1D `w` ↔ Linear `weight` + add/drop layers" remapping.

## Key points and pitfalls

1. **Definition of missing/unexpected**: missing = present in the new model but not in the (post-mapping) checkpoint → `blocks.{0,1}.gate`; unexpected = present in the checkpoint but not in the new model → `lm_head.weight`. The judgment must happen **after** remapping; otherwise all 7 renamed keys would show up in both lists at once.
2. **Return order**: the judge pins `PYTHONHASHSEED=0`, but `sorted` is still the only reliable alignment; the raw `_IncompatibleKeys` lists make no ordering promise, so the statement requires sorting yourself.
3. **Value transform and key transform are separate concerns**: `embed.w → embed.weight` needs transposition in addition to renaming; the (3,2) \([[25,26],[27,28],[29,30]]\) flattened row-major after transposing is 25,27,29,26,28,30. Renaming without transposing makes `load_state_dict` raise `RuntimeError` on the shape mismatch (strict=False does not rescue shape errors).
4. **Unloaded parameters keep their initialization**: the two `gate` layers are new; after a strict=False load they remain zero-initialized — this is exactly the "missing ≠ error, it is incremental initialization" semantics.
5. **Same-shape-but-different-semantics keys go undetected**: strict=False only checks names and shapes, not semantics; real conversion scripts must guarantee mapping correctness themselves (this problem checks it through the value ledger).

## Cross-check basis

The solution uses the real `load_state_dict`; the judge's `_oracle` never touches `nn.Module`: it hand-derives the 8 checkpoint value segments from the stated numbering formula, writes missing/unexpected by hand using a hardcoded old→new table plus set differences, spells out the embed transposition by index, and compares `loaded` exactly against the hand-assembled table (small integers are lossless in float32, `torch.equal` level). Three degenerate assertions are added: gates stay zero, embed is not a verbatim copy, and the two key lists are sorted.

## Complexity

The whole flow is \(O(\text{total parameters})\) (about 40 elements) and instantaneous; the difficulty lies purely in API semantics.
