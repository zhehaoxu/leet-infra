# notes (ag-voyager-skill-lib)

## Source

Official implementation `MineDojo/Voyager`, file `voyager/agents/skill.py` (127 lines), read end to
end and checked line by line:

- **Skip gate** (L62-64): `if info["task"].startswith("Deposit useless items into the chest at"): return`
  — the whole `add_new_skill` returns early: no dump, no library entry, and **no record at all**
  (this problem adds a `skipped` marker purely for accounting).
- **Same-name overwrite** (L71-79): `if program_name in self.skills:` then
  `self.vectordb._collection.delete(ids=[program_name])`, then
  `i = 2; while f"{program_name}V{i}.js" in os.listdir(...): i += 1`, dump name `f"{program_name}V{i}"`;
  otherwise (L78-79) `dumped_program_name = program_name`.
- **Write order** (L80-88): `add_texts(ids=[program_name])` happens *before*
  `self.skills[program_name] = {...}`. `self.skills` is a **dict**, so assigning an existing key keeps
  its original insertion position — that is where "overwrite does not move `order`" comes from.
- **Where `synced` comes from** (L45-50 and L89-91): two
  `assert self.vectordb._collection.count() == len(self.skills)`. The overwrite branch deletes the
  same id at L73 before re-adding it, so the count never grows — the assertion holds on every legal path.
- **Description template** (L102-112): `skill_description = f"    // {self.llm(messages).content}"`,
  then `return f"async function {program_name}(bot) {{\n{skill_description}\n}}"`.
  Note the two-layer prefix (four spaces plus `// `); the problem folds them into one template.
- **Retrieval** (L114-127): `k = min(self.vectordb._collection.count(), self.retrieval_top_k)`,
  `if k == 0: return []`, then `similarity_search_with_score(query, k=k)` and codes taken in the
  returned order via `self.skills[doc.metadata["name"]]["code"]`.
- **programs** (L52-59): first every `self.skills.items()` entry as `f"{entry['code']}\n\n"`, then all
  control primitives, also with `\n\n`.
- Paper arXiv 2305.16291 describes the three layers (skill library, automatic curriculum, iterative
  prompting), but dump/overwrite/retrieval details live in `skill.py`; the code wins.

## Normalization in this problem

- **Vector store replaced by an explicit distance table**: a `retrieve` event carries `distances` whose
  length must equal the current skill count; ordering is stated as "distance ascending, ties by
  insertion order", mirroring Chroma's ordered result.
- **Dump IO replaced by a `dump_files` set**: only the semantics "V-number probing looks at which files
  already exist in the directory" survive, nothing is written; `dumps[i]["file"]` is the base name
  without `.js` (the official code also computes the base name first and appends the suffix).
- **LLM description generation replaced by an input**: `llm_description` is
  `self.llm(messages).content`; the template wrapping still happens inside the function, because that
  is what this problem tests.
- **`skills.json` persistence and `vectordb.persist()` dropped**: pure IO, no effect on the ledger.
- **Return type of `retrieve_skills`**: the official method returns only `codes`; this problem also
  records `names` and `k`, because "how many were taken" and "in what order" are the core criteria here.
- **`synced` kept**: it mirrors the two official `assert`s. Since the overwrite branch deletes before
  adding, the field is `True` for every legal input and is only an alignment note about
  "vector store in sync with the dict" — it is **not** a criterion (see the equivalent mutant below).

## Mutants

27 mutants, 26 CAUGHT, 1 proven equivalent:

```
m-guard-short             relax prefix to "Deposit"            -> check_guard_prefix
m-guard-lower             lowercase the prefix comparison      -> check_guard_prefix
m-guard-fallthrough       skip but do not continue             -> check_guard_prefix
m-skip-no-name            skipped drops program_name           -> check_guard_prefix
m-rewrite-false           treat overwrite as append            -> check_same_name_overwrites
m-version-start-1         start V numbering at 1               -> check_version_suffix_scan
m-version-no-scan         no free-slot probing                 -> check_version_suffix_scan
m-dump-no-ext             forget the .js when registering     -> check_version_suffix_scan
m-new-skill-versioned     version numbers on new skills too    -> check_version_suffix_scan
m-new-skill-no-order      new skill never enters order         -> several checks
m-book-raw-llm            description not wrapped              -> check_description_template
m-desc-indent2            comment indent reduced to two spaces -> check_description_template
m-desc-no-comment         comment loses the "// "             -> check_description_template
m-k-no-min                k no longer clamped to library size  -> check_retrieve_top_k
m-tie-late-first          ties resolved last-in-first          -> check_retrieve_tie_by_insertion
m-take-tail               take the tail k instead of the head  -> check_retrieve_top_k
m-dist-len-off            distances length unchecked          -> check_validation
m-dist-bool-ok            distances accept bool                -> check_validation
m-topk-bool-ok            top_k accepts bool                   -> check_validation
m-topk-min-zero           top_k allows 0                       -> check_validation
m-prims-first             primitives placed before skills      -> check_programs_concatenation
m-skill-sep-one           skill separator loses a newline      -> check_programs_concatenation
m-prim-sep-one            primitive separator loses a newline  -> check_programs_concatenation
m-dup-name-ok             duplicate names not rejected         -> check_validation
m-unknown-kind-ok         unknown event kind let through       -> check_validation
m-count-dumps             count counts dumps instead of skills -> check_same_name_overwrites
m-synced-true             synced constant True                 -> equivalent (below)
```

**Proof of equivalence**: `synced` is defined as `len(book) == len(order)`. Both grow only in the
"new skill" branch (L86-87 put `dumped = name` and `order.append(name)` in the same `else`), neither
moves in the overwrite branch, both start at the same length given by `skills`, and duplicate names
inside `skills` are rejected beforehand. So `len(book) == len(order)` holds for every legal call and the
constant `True` is indistinguishable from the original — not a coverage gap, but a consequence of the
official design where L73 deletes the id before L80 adds it back (count unchanged).

## Pitfalls

- The first version of all 10 checks was written as `def check_xxx():` with no parameter, which raises
  `TypeError: takes 0 positional arguments but 1 was given`; both the harness and selftest pass `solve`,
  so the parameter is mandatory.
- In `check_against_oracle` the expected `dumps` order was first written as the overwrite first; the real
  order follows the **event sequence**, not the name order.
- The "new skills get V numbers too" mutant is only caught by the `craft` case with
  `dump_files=["craft.js","craftV2.js"]` and no `craft` in the library: the first dump must be plain
  `craft`. Testing only the "library already has that name" scenario would miss it.
- The last block of `programs` also carries `"\n\n"`, so a non-empty result always ends with two
  newlines; do not `rstrip` it in assertions.

## Overlap check

- `ag-generative-memory-retrieval`: also a "top-k retrieval by query", but over dialogue memories with
  similarity thresholds and scoring; no name overwrite, no versioned dump, no prompt assembly.
- `ag-memorybank-forget-ledger`: forgetting curves and eviction in a memory bank; entries only shrink,
  never overwritten.
- `ag-ace-skillbook-ledger`(750): also a "skill book", but that one applies delta operations to a policy
  document and renders it; no file naming and no vector retrieval.
- `ag-funsearch-island-ledger`(748), `ag-dgm-archive-ledger`(749): evolutionary pools over a program
  library, entries are never overwritten and there is no top-k retrieval.
- `peft-trainable-tokens-replace` only shares the word token; the mechanism is unrelated.
