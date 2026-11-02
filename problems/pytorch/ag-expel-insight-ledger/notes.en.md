# notes (ag-expel-insight-ledger)

## Source

Official implementation `LeapLabTHU/ExpeL`; two files were checked:

- `agent/expel.py` (743 lines)
  - `parse_rules`(L665-681): the regex at L666 is
    `((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)`; the three gates at L673
    (non-empty + contains none of `['ADD','AGREE','EDIT']` + `endswith('.')`); the normalization
    `if 'ADD' in operation` at L677.
  - `update_rules`(L696-743): the first pass (L698-719) only rewrites `operations` (ADD dedupe, EDIT
    turned into `AGREE` when it hits an existing rule, out-of-range numbers and unmatched texts pushed
    into `delete_indices`); the second pass (L721-739) applies them in the fixed order
    `['REMOVE','AGREE','EDIT','ADD']`; L730 `remove_strength = 3 if list_full else 1`; L740 filters
    `> 0`; L741 `sort(key=count, reverse=True)`.
  - Full-list test at L314: `list_full = self.max_num_rules + 5 <= len(self.rule_items_with_count)`,
    evaluated **before** the batch is applied.
  - Rendering at L399: `'\n'.join([f"{i}. {item}" for i, item in enumerate(self.rule_items, 1)])`;
    note it renders the **plain-text** list (counts only go to the log).
  - `retrieve_rule_index`(L683-687) / `is_existing_rule`(L689-693): the match direction is
    `rules[i][0] in operation_rule_text`, i.e. "old text is a substring of new text", returning the
    **first** hit.
- `insight_extraction.py` (169 lines): L144 `react_agent.create_rules(...)`, L156 the field whitelist for
  checkpointing. All three critique sources (failure-only-when-more-than-one at L177, success chunking by
  `success_critique_num` at L159, the 13000-token truncation with three strategies at L179) live on the
  `expel.py` side and are orthogonal to this ledger.

## Normalization in this problem

- **LLM output becomes an explicit input**: `rounds` is exactly the per-batch `llm_output` text; no model
  is called. The `parse_rules` regex and its three gates are preserved verbatim because they are one of
  the main points of the problem.
- **Trajectory pairing and the critique gates are dropped entirely**: the success-by-failure cross
  product, "only reflect when failures > 1", the `random_divide_list` random chunking, and the `random`
  branch of the 13000-token truncation are either randomness-dominated or need a real token counter.
  The problem starts from "the LLM output already exists".
- **embedding / vectorstore / fewshot retrieval dropped**: ExpeL's top-k only happens on fewshot
  trajectory retrieval (`setup_vectorstore` + `topk_docs`); rules are injected **in full**
  (`insert_before_task_prompt` stuffs `self.rules`). So the insight ledger itself has no kNN, and this
  problem deliberately adds none, avoiding a collision with `ag-generative-memory-retrieval`.
- **Checkpoint resume dropped**: the whole `loaded_dict`/`loaded_log`/`resume_flag` machinery exists for
  long-running jobs; the ledger does not depend on it.
- **`max_num_rules` is pinned to "only affects REMOVE strength"**: the official code never truncates by
  it, and both the statement and the hints say so.

## Mutants

29 mutants, 28 CAUGHT, 1 proven equivalent:

```
m-regex-no-label          stop swallowing the "label: " prefix -> parse_prefix_label
m-regex-no-number         verbs no longer accept a number        -> parse cases (EDIT 3 stops matching)
m-regex-anchored          anchor the regex with ^               -> midword case (PADD stops matching)
m-banned-empty            empty banned-word list                 -> three_gates
m-banned-remove           add REMOVE to the banned list          -> three_gates
m-no-period-gate          drop the trailing-period requirement   -> three_gates
m-banned-substring        banned check becomes whole-string equal -> three_gates
m-add-number-kept         keep the number on ADD                 -> add_parses_into_count_two
m-int-bool                integer validation accepts bool        -> validation
m-match-reversed          reverse the containment direction      -> add_dedupes_by_substring
m-add-dedupe-inverted     invert the ADD dedupe condition        -> add_dedupes_by_substring
m-edit-agree-index        AGREE index off by one                 -> edit_becomes_agree
m-edit-num-off            relax the number upper bound by one    -> edit_invalid_number
m-remove-exist-inverted   invert REMOVE/AGREE existence check    -> remove_unknown_text
m-remove-strength-one     REMOVE always -1                       -> remove_strength
m-remove-strength-three   REMOVE always -3                       -> remove_strength
m-agree-double            AGREE adds 2                           -> prune_and_stable_sort
m-edit-no-bump            EDIT does not bump the counter         -> edit_by_number
m-edit-index-zero         EDIT always rewrites index 0           -> edit_by_number
m-add-count-one           ADD starts at 1                        -> add_parses_into_count_two
m-order-edit-first        apply EDIT before AGREE                -> apply_order case (see below)
m-prune-zero              keep rules at count 0                  -> prune_and_stable_sort
m-sort-asc                reverse the sort direction             -> prune_and_stable_sort
m-full-threshold          full-list test becomes `max <= len`    -> remove_strength
m-prompt-off-by-one       rendering starts numbering at 2        -> render_prompt
m-count-log               count reports rounds instead of rules  -> add_dedupes / render
m-log-applied-from-parsed log `applied` filled with parsed ops   -> edit_becomes_agree
m-listfull-const          list_full constant True                -> remove_strength
m-min-zero                `value < minimum` -> `value < 0`       -> equivalent (below)
```

**Proof of equivalence**: `_as_int` has only two call sites in this problem, and the only one passing
`minimum` is `max_num_rules`, which passes `0`. Hence `value < minimum` and `value < 0` agree value by
value there (`-1` triggers both, `0` passes both). Not a coverage gap: deleting the `minimum` parameter
and writing `value < 0` inline would still pass every check.

## Pitfalls

- **The trailing period makes containment rare**: every rule ends with `.`, so "old text is a substring of
  new text" only holds when the old rule appears as a complete sentence (period included) inside the new
  one, e.g. old `Check the drawer.` inside new `Check the drawer. Then open it.` The first version of the
  dedupe case used old `open the drawer.` with new `open the drawer before checking.`; the period cut the
  match, so the "duplicate" was never detected and the case itself was wrong.
- `list_full` needs `len >= max_num_rules + 5`; since `max_num_rules` bottoms out at 0, at least 5 rules
  are required to trigger the -3. The mixed case never reaches 5 rules, so a separate 5-rule setup is
  needed.
- `m-order-edit-first` is caught by a **crash**: with the order swapped, AGREE still locates by text while
  EDIT has already replaced that text, so `retrieve_rule_index` returns `None` and `rules[None]` raises
  `TypeError`. That is exactly why the official `# Order is important` comment exists.
- Sort stability: `sort(key=count, reverse=True)` does **not** reverse equal-count elements; the oracle
  writes `sort(key=lambda r: -r[1])` (ascending on the negated key) and both give the same order for ties.
- The first version of `check_against_oracle` expected the second round's `EDIT 3` to land; in reality the
  number is out of range and the same round's ADD is dropped as well (its body contains an existing rule),
  so the whole round is a no-op — which then became part of the assertions.

## Overlap check

- `ag-ace-skillbook-ledger`(750): also "the LLM emits operations and we fold them into a text book", but
  there entries carry ids and are addressed by id, with keyword normalization, section folding,
  sign-only `delta`, and soft delete / purge. Here entries have **no id**; addressing is "old text is a
  substring of new text", counters are signed integers (ADD=2, AGREE=+1, REMOVE=-1/-3, EDIT=+1), and the
  table is pruned and sorted by count. Not a single criterion overlaps.
- `ag-voyager-skill-lib`(751): same-name overwrite, versioned dump, top-k retrieval; no text matching and
  no counters.
- `ag-funsearch-island-ledger`(748), `ag-dgm-archive-ledger`(749): evolutionary sampling over program
  pools and archives; entries only grow and are never pruned by count.
- `ag-reflexion-memory-ledger`: writes and retrieves reflection entries; no operation batches and no
  counter-driven eviction.
- `ag-memgpt-paging-ledger`, `ag-zep-invalidation-ledger`: page swapping and invalidation propagation,
  unrelated mechanisms.
