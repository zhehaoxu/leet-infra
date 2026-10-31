# notes — ag-tool-loop-breaker (order 667)

## Source

- Problem family: agent-runtime tool-loop guards (loop guards). Upstream frameworks fingerprint `(tool, canonical(args))` and trip on the occurrence count (cache semantics plus a counting gate), e.g. step budget / duplicate-call detection; this problem pins the ledger abstractly without binding to a specific framework.
- Two canonicalization decisions are observable contracts:
  - `sort_keys=True`: JSON object key order does not affect the fingerprint (the same semantic call must not drift with dict construction order);
  - `float round 6`: model-generated float jitter at the 1e-7 level must not create a new fingerprint, while 1e-5-level differences count as distinct calls.
- Decision semantics: the **pre-call count** `c` (counts_at_call) is compared as `c < threshold`; the count increments only on allow; threshold=0 blocks at step one.

## Judging

- Oracle: `test_cases._py` is an independent rewrite (Counter counting, a separate sorted-keys recursion path, list-comprehension first-block scan), 150 random cross-checks (n 1–10, threshold 0–3, args depth 2, tool pool includes the `Read`/`read` case pair).
- Anchors: one fingerprint over three steps at threshold=2/3/0; independent counting for distinct tools/args; key-order irrelevance (nested included); float 1e-7 same fingerprint vs 1e-5 different; multiple blocks keep the first; fingerprint format `tool|` + 64 hex chars (checked against a hand-computed sha256).
- Invalid inputs: empty calls, missing/extra keys, empty tool, non-str tool, non-dict args, threshold bool/negative/float/str → ValueError.

## Mutation log (9/9 caught)

threshold-off-by-one (`<` → `<=`), no-sort-keys, no-float-round, tool-lower (case merging; caught by the Read/read random pool), sha1-instead, no-count-increment (always 1, never trips), first-block-last (records only the final step), separators-default (json default separators), allow-threshold-zero (threshold 0 lets everything through).

## Red lines

No real tool execution, network calls, or framework dependencies; boundary with `agent-react-loop` (loop state machine): this is a **parameter-level fingerprint ledger**, not ReAct step semantics.
