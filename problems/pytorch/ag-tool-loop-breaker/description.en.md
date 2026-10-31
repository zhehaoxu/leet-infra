# Tool-Loop Breaker: Fingerprint Dedup Ledger

## Problem

A common failure mode of LLM agent runtimes is tool-call loops (repeated calls with identical arguments). A standard guard is a **fingerprint dedup counter**: fingerprint every call, and trip a breaker once the same fingerprint exceeds a threshold. Given a step-by-step call list `calls` (each `{"tool", "args"}`) and a `threshold`, produce the per-step ledger: whether the call is allowed, its fingerprint, how many times that fingerprint had already appeared before this call, and the index of the first blocked step.

## Solution

The fingerprint ledger is pinned by the spec:

1. Canonicalize `args`: recursively round every `float` to 6 decimals (`round(v, 6)`; test `bool` before `int`), then `json.dumps(..., sort_keys=True, separators=(',', ':'))` — key order is irrelevant and 1e-7-level float jitter maps to the same fingerprint;
2. `fingerprint = tool + "|" + sha256(canonical).hexdigest()` (64 hex chars).

Decision ledger: let `c` be the number of prior occurrences of this fingerprint (recorded in `counts_at_call`); if `c < threshold` the call is allowed and the count is incremented, otherwise it is blocked (`allowed=False`, count unchanged). With `threshold=0` the first step blocks. `first_block_index` is the first blocked step (0-based), or `None` if nothing was ever blocked. Returns `{"allowed", "fingerprints", "counts_at_call", "first_block_index"}`.

## Constraints

- 1 ≤ len(calls) ≤ 10^4; args may contain float/int/str/bool/None and nested list/dict.
- Empty/non-list calls, entries with missing or extra keys, tool not a non-empty str, args not a dict, threshold bool/negative/non-int → `ValueError`.
