# Graph of Thoughts Operations Ledger (ag-got-operations-ledger)

**Difficulty**: Medium | **Collection**: agentic-systems | **Order**: 723

Stateless-ify the `operations.py` / `thought.py` of the official Graph of Thoughts (GoT, arXiv 2308.09687) framework [spcl/graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) over a **linear operation chain**: the Thought setter flag ledger and the deterministic scheduling semantics of all nine operations (score / validate_and_improve / generate / improve / aggregate / keep_best_n / keep_valid / ground_truth / selector), each accounted step by step. All LLM calls are made explicit as inputs (scores/valids/updates/parsed/results sequences); prompter/parser/lm are not modeled.

## What you implement

`solve(ops, base_state=None)`: execute the operation descriptor list in order, returning a per-step ledger and the final thoughts.

## Ledger points (all verified line-by-line against the official source)

1. **Thought setter flags**: the `score` setter sets `scored=True`, the `valid` setter sets `validated=True`, the `solved` setter sets `compared_to_ground_truth=True` (even when the result is False). Defaults: `score=0.0`, all flags False.
2. **Two kinds of thought provenance**: `Score`/`Selector`/`KeepBestN`/`KeepValid`/`GroundTruth`/`ValidateAndImprove`'s initial attempt go through `from_thought` (copies all flags); `Generate`/`Improve`/`Aggregate` and ValidateAndImprove's **improved products** go through `Thought({**state, **update})` — a **brand-new default thought whose score resets to 0.0 and all flags reset**.
3. **Score combined's zip truncation**: combined mode zips scores with thoughts — when scores is shorter, the extra thoughts are **silently dropped**; individual mode scores each thought.
4. **ValidateAndImprove's attempt ledger**: each round consumes one boolean (recorded first, break checked after); with `improve=True` and a persistently invalid thought there are exactly `num_tries+1` validations and `num_tries` improvements (the break condition `current_try >= num_tries` counts **after** the improvement). The last attempt in the chain is the final thought.
5. **Generate's idle skip and merging**: a predecessor that exists but has zero thoughts → skip outright (cascading skips); new states merge as `{**parent, **ns}` (parent keys kept, new keys override); producing more than `p*r*parents` is only an official log warning, **no truncation**.
6. **Aggregate's score-ordered merge**: base_state is built by merging each predecessor in **ascending score order** (stable, ties keep original order) via `{**base, **state}` — **higher scores merge later and override same-name keys** (official comment "applied in order of score"); unscored thoughts use the default 0.0. A dict parsed is wrapped into a single-element list.
7. **KeepBestN's two accounting regimes**: all predecessors must be scored (else ValueError); sort by score (stable, ties keep original index order) and take the first n. If sorting fails (e.g. scored but score=None), the official fallback keeps **only thoughts with `isinstance(score, float)`** — int scores are dropped too. `n<=0` raises ValueError at construction.
8. **KeepValid's pass-through quirk**: the keep condition is `not validated or valid` — **unvalidated thoughts pass through**; only "validated and invalid" ones are dropped.
9. **GroundTruth's exception swallowing**: an evaluation exception → `solved=False` (exception type irrelevant).
10. **Domain constraints**: calling score/validate_and_improve/improve/aggregate/keep_best_n/keep_valid/ground_truth as the first operation (no predecessor) raises ValueError (official asserts converted to ValueError).

## Input conventions

- Operation descriptors as documented in the starter comment; the `scores`/`valids`/`updates`/`parsed`/`results` sequences are given in consumption order and long enough for each case's path (except combined-score truncation cases, deliberately short).
- No LLM in the judging environment: all generation/validation/scoring results are materialized as inputs.

## Example

```python
ops = [
    {"op": "generate", "parsed": [[{"v": "a"}, {"v": "b"}]]},
    {"op": "score", "combined": False, "scores": [1.0, 9.0]},
    {"op": "aggregate", "parsed": {"z": 0}},
]
led = solve(ops)
# steps[2]["base_state"] == {"v": "b"}   # 9.0 merged later, overrides 1.0
# steps[2]["thoughts"][0]["state"] == {"v": "b", "z": 0}
# steps[2]["thoughts"][0]["score"] == 0.0  # aggregate product is brand-new
```

## Scoring

13 checkpoints: zip truncation, setter flags, the num_tries+1 attempt ledger, generate skip/merge, aggregate ordered override, KeepBestN ties and the fallback (including int-score dropping), KeepValid pass-through, GroundTruth exceptions, selector flag copying, improve resets, domain constraints, plus 100 randomized operation-chain cross-checks.
