# notes (ag-ace-skillbook-ledger)

## Sources

- Official implementation `kayba-ai/agentic-context-engine`, `ace/core/skillbook.py` (899 lines);
  `ace/deduplication/operations.py` was checked too but is not used here.
  - `_normalize_optional_text` / `_normalize_required_text`: `str()` plus `strip()`, empty becomes
    `None`.
  - `_normalize_keyword`: `strip` then `re.sub(r"\s+", "_", text.lower())`; `_normalize_keywords`
    deduplicates by first occurrence with a `seen` set.
  - `_coerce_section_and_keywords`: a legal section passes through; otherwise keywords become
    `normalize([section] + keywords)` and the section folds to `DEFAULT_LEGACY_SECTION = "context"`.
  - `add_skill`: `if not normalized_keywords: normalized_keywords = _normalize_keywords([raw_section])`;
    `if section == "context" and insight is None:` then splits into "raw section illegal -> insight =
    issue" and "legal -> raise"; `self._sections.setdefault(section, []).append(skill_id)` appends
    unconditionally, so a repeated id is registered twice.
  - `update_skill`: `issue`/`keywords`/`insight` use the module-level `_UNSET` sentinel to distinguish
    "leave alone"; normalized keywords that come out empty raise
    `ValueError("keywords are required and must be non-empty")`; afterwards
    `if skill.section == "context" and not skill.insight: raise`.
  - `_apply_operation`: UPDATE passes `operation.keywords if operation.keywords else _UNSET`, so an
    **empty list means "leave alone"**; TAG does `delta = int(metadata.get("delta", 0))` and clamps it
    to ±1 by sign.
  - `tag_skill` / `mark_used` / `remove_skill(soft=True)` / `purge` / `as_prompt` / `stats` are mirrored
    line by line.
- Paper: arXiv 2510.04618 (Agentic Context Engineering). It describes the generator/reflector/curator
  triad; details such as the helpful/harmful counters and the render order follow `skillbook.py`.

## Normalizations

- **Clock and embedding dropped**: `created_at` / `updated_at` / `embedding` are not part of the ledger
  (`update_skill` resets the embedding, which is unobservable side state).
- **Id generation**: the official code uses an internal `_generate_id`; here it is normalized to
  `"<section>-<n>"` (n = current entry count + 1), only used when `skill_id` is omitted.
- **`insight_source` / `occurrences` / `similarity_decision` dropped entirely**: that is provenance and
  "are these two similar entries merged" bookkeeping, orthogonal to delta application, and left for a
  later problem.
- **Mid-batch exceptions in `apply_update`**: the official code does not catch, so one bad operation
  aborts the batch with the earlier ones already applied. This problem also raises `ValueError`
  directly (the partially applied state is unobservable, so it is not modelled).
- **`MARK_USED`** is not a model-produced delta; it is the framework's `mark_used(skill_ids)`. To make
  it judgeable it is promoted to an operation type, and the statement says so.

## Mutations

30 mutations, all caught:

```
m-text-no-strip          no strip                                  → normalization case
m-kw-no-lower            keywords not lowercased                   → normalization case
m-kw-no-ws               `\s+` not turned into underscore          → normalization case
m-kw-no-dedupe           keywords not deduplicated                 → normalization case
m-legacy-no-fold         legacy section not folded                 → folding case
m-legacy-no-kw           legacy section name not a keyword         → folding case
m-legacy-no-insight      legacy section does not fall back to issue → folding case
m-context-no-raise       context without insight does not raise    → folding case
m-default-kw             empty keywords not defaulted to [section] → ADD case
m-update-empty-kw-sets   empty list treated as "set empty"         → UPDATE case
m-update-insight-unset   empty-string insight no longer clears     → empty-string case
m-update-no-context-check  post-update context check removed       → clearing case
m-tag-delta-raw          delta accumulated by raw value            → counter case
m-tag-zero-helpful       neutral counted as helpful                → counter case
m-remove-soft-noop       soft delete does not clear active         → soft-delete case
m-purge-not              purge degrades to soft delete             → soft/purge case
m-purge-keep-bucket      purge does not clean the section list     → section assertion
m-used-any-active        inactive entries also bumped              → counter case
m-used-no-bump           used not incremented                      → counter case
m-prompt-order           the two sections render in reverse order  → render case
m-prompt-empty-section   empty section still emits its heading     → render case
m-prompt-insight-always  insight always rendered (even None)       → render case
m-prompt-no-strip        trailing blank lines kept                 → render case
m-stats-total-active     skills counts only active entries         → stats case
m-stats-sections-sum     sections counts section keys              → stats case
m-add-dup-guard          duplicate id no longer re-registered      → duplicate-ADD case
m-op-unknown             unknown operation type accepted           → validation case
m-delta-bool             delta accepts bool                        → validation case
m-skill-nonneg           counters no longer require non-negative   → validation case
m-markused-any-seq       skill_ids accepts a string                → validation case
```

## Pitfalls

- The first attempt used `insight: "  "` to catch "empty string clears"; but `"  "` is **truthy**, so
  `if insight:` took the branch too and the mutant survived. Only a real `""` separates "leave alone"
  from "clear".
- A bare `{"type": "NOPE"}` does "catch" the unknown-type mutant, but for the wrong reason (it falls to
  the `else` branch, `skill_ids` is missing, and that raises). Verifying the whitelist needs an unknown
  operation with **all** fields present.
- `stats["sections"]` and the number of keys in `stats["by_section"]` are different quantities: the
  former counts non-zero sections, the latter always carries both keys.

## Duplicate checks

- `ag-memgpt-paging-ledger` (page-in/page-out) and `ag-zep-invalidation-ledger` (invalidation
  propagation): both are entry lifecycles, but by paging and invalidation, not delta batches with
  counters plus rendering.
- `ag-reflexion-memory-ledger`(725): reflection writes and retrieval, without `_UNSET` sentinels, soft
  delete or section folding.
- `ag-self-instruct-filter`(747): filtering and dedup over candidate instructions, entries only grow.
- `ag-funsearch-island-ledger`(748), `ag-dgm-archive-ledger`(749): evolutionary pool management for
  programs/archives, orthogonal to rewriting a strategy document through incremental operations.
