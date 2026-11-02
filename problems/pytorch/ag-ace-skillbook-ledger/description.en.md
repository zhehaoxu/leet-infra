# ACE Skillbook: Delta Batch Application and Prompt Rendering Ledger

ACE treats the context itself as an evolving strategy document: the Generator works, the Reflector
distils insights, and the Curator folds them into a batch of **delta operations** (add, update, tag
helpful/harmful, remove) applied to the skillbook; the next round renders the whole book into the
prompt. `kayba-ai/agentic-context-engine`'s `skillbook.py` is that book, and this problem turns it into
a pure function ledger.

## Interface

```python
def solve(skills, operations) -> dict
```

`skills` is the initial entries in insertion order, each
`{"id", "section", "keywords", "issue", "insight", "active", "used", "helpful", "harmful", "neutral"}`;
`operations` is the batch, each shaped like
`{"type": "ADD"|"UPDATE"|"TAG"|"REMOVE"|"MARK_USED", "section", "issue", "keywords",
"insight", "skill_id", "delta", "purge", "skill_ids"}`.

The result carries `order`, `skills` (id to entry, soft-deleted ones included), `sections` (section to
id list, duplicates allowed), `active`, `prompt`, `stats` and `log`.

## Input and Output

```
Input:  skills=[]
     operations=[{type:"ADD", section:"Tooling", issue:"prefer ripgrep",
                  keywords:["Search"], skill_id:"l1"}]
Output: skills["l1"].section   = "context"          # folded, not context/harness
     skills["l1"].keywords  = ["tooling","search"]
     skills["l1"].insight   = "prefer ripgrep"    # falls back to issue
     sections              = {"context": ["l1"]}
```

## Rules

- **Text normalization**: everything goes through `str()` then `strip()`, and an empty result becomes
  `None`; `issue` and `section` must be non-empty or `ValueError` is raised.
- **Keyword normalization**: `strip` → `lower` → `\s+` becomes `_`, empties dropped, **deduplicated by
  first occurrence**.
- **Section folding**: only `context` / `harness` are legal (case-insensitive); any other value folds to
  `context`, the **original section name** (normalized) is prepended to the keywords, and `insight`
  falls back to `issue` when absent. A real `context` entry without `insight` raises; `harness` does not
  care.
- **ADD**: empty keywords default to `[section]`. Re-ADDing the same `skill_id` overwrites the record
  but **appends the id to the section list again** (so it renders twice). Without `skill_id` the id is
  generated as `"<section>-<n>"`.
- **UPDATE**: `issue` / `insight` set to `None` means "leave alone"; an **empty list** of `keywords`
  also means "leave alone", while a non-empty list of blanks raises. After the update a `context` entry
  must still have an `insight`. A missing `skill_id` or one that points nowhere logs `noop`.
- **TAG**: `delta` only contributes its **sign** (`7` counts like `1`), and `0` increments `neutral`.
- **REMOVE**: soft delete by default (`active=False`, record kept); `purge=True` really deletes it,
  removes the id from the section list and drops the key when the section empties.
- **MARK_USED**: only `active` entries gain `used`; the log lists just the ids that were bumped.
- **Rendering**: fixed `context` → `harness` order; each section starts with `## <section>`, each entry
  emits `- [<id>]`, `  Keywords: a, b`, `  Issue: ...`, and **only when `insight` is non-empty** an
  extra `  Insight: ...`; every entry is followed by a blank line and the whole string is `rstrip()`ed.
  An empty book renders as an empty string.
- **stats**: `sections` counts sections **with active entries** (not section keys), `skills` counts
  **including soft-deleted** records, `active_skills` counts only active ones, and `by_section` always
  carries both keys starting at 0.
- **Validation**: unique entry ids, legal `section`, non-negative integer counters (`bool` rejected);
  unknown operation types, non-integer `delta` and non-sequence `skill_ids` all raise `ValueError`.

## Hints

- "Empty list = leave alone" and "empty string = clear" are two **different** sentinels:
  `keywords: []` keeps the old keywords, while `insight: ""` really clears a harness entry's insight to
  `None`.
- A soft-deleted entry still counts in `stats["skills"]`, but it is not in `active`, does not render and
  is not bumped by `MARK_USED`.
- Re-ADDing the same id makes the section list hold a duplicate id, so the same entry prints twice —
  official behaviour, not an artefact of this problem.
