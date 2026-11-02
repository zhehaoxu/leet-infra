# Voyager Skill Library: Name Overwrite, Versioned Dump and Top-k Retrieval

Voyager (arXiv 2305.16291) lets an LLM player keep writing new skills inside Minecraft — a small
`async function` snippet in JS — and pushes each one into a skill library. Before the next task it
retrieves the few most relevant snippets and puts them into the prompt. `voyager/agents/skill.py`
in `MineDojo/Voyager` is exactly that library; this problem turns it into a pure-function ledger.

## Interface

```python
def solve(skills, events, retrieval_top_k=5, dump_files=None, primitives=None) -> dict
```

- `skills`: initial library content, `[{"name", "code", "description"}]`, in **insertion order**.
- `events`: events in chronological order, two shapes:
  `{"kind": "add", "task", "program_name", "program_code", "llm_description"}`
  and `{"kind": "retrieve", "query", "distances"}` (`distances` aligns with the **current** skills).
- `dump_files`: file names already present in the dump directory (with extension).
- `primitives`: control-primitive code blocks, appended after the skills when building the prompt.

Returns `order` (insertion order of skill names), `skills` (name -> `{code, description}`),
`dumps` (one record per write), `retrieved` (one record per query), `skipped`, `programs`
(the concatenated code string), `count`, `synced`.

## Input / Output

```
Input:  skills=[{name:"mine", code:"M1", description:"..."}]
        events=[{kind:"add", task:"Deposit useless items into the chest at 0,64,0",
                 program_name:"deposit", ...},
                {kind:"add", task:"Craft a wooden pickaxe", program_name:"craft", ...},
                {kind:"add", task:"Mine iron", program_name:"mine",
                 program_code:"M2", llm_description:"mine iron"},
                {kind:"retrieve", query:"dig", distances:[0.9, 0.1]}]
        retrieval_top_k=2, dump_files=["mine.js"]
Output: order     = ["mine", "craft"]      # overwrite keeps position, new names append
        dumps     = [{"event":1, "program_name":"craft", "file":"craft", "rewritten":False},
                     {"event":2, "program_name":"mine",  "file":"mineV2","rewritten":True}]
        skipped   = [{"event":0, "program_name":"deposit"}]
        retrieved = [{"event":3, "query":"dig", "k":2,
                      "names":["craft","mine"], "codes":[...]}]
```

## Rules

- **Skip gate**: if `task` starts with `Deposit useless items into the chest at`, the whole event is
  dropped and only a marker lands in `skipped` — no dump, no library change, no new name. The prefix
  must match **verbatim** (case, spaces and the trailing `at` all count).
- **Same-name overwrite**: when `program_name` already exists, the entry's content is replaced but
  its **position is unchanged** and `count` stays the same. The dump name is no longer the original
  name: scan `"<name>V<i>.js"` from `i = 2` upward and take the first index whose file is **not** in
  `dump_files` (with `mine.js` and `mineV2.js` both taken, the answer is `mineV3`).
- **New skill dump**: uses `program_name` as the file name with **no** V-number probing — even if
  `craft.js` and `craftV2.js` already exist.
- **Description template**: the skill description is not the raw LLM text but
  `async function <name>(bot) {\n    // <llm_description>\n}` — note the four-space indent and the
  closing brace.
- **Retrieval size**: `k = min(current skill count, retrieval_top_k)`; with an empty library `k = 0`
  and an empty result is returned without looking at `distances`. Ordering is "distance ascending,
  ties by insertion order", take the first `k`.
- **programs concatenation**: all skill code in insertion order, then the control primitives, with
  `"\n\n"` after **every** block; all-empty yields the empty string.
- **Validation**: duplicate names in `skills`, a non-positive-integer `retrieval_top_k`, a `distances`
  length that does not match the current skill count, an unknown event kind, or a `bool` inside
  `distances` all raise `ValueError`.

## Hints

- "Overwrite" and "append" are indistinguishable from `count` alone; only whether `order` grew tells
  them apart. This is the easiest place to get wrong.
- V-number probing looks only at **dump file names**, never at whether the library already holds that
  name: a directory that already has `mineV2.js` but no `mine` in the library still dumps the first
  `mine` as `mine`, because it takes the "new skill" branch.
- The last block of `programs` also carries `"\n\n"`, so a non-empty result always ends with two
  newlines — do not `rstrip` it.
