# ExpeL Insight Ledger: Rule Op Parsing, Counter Updates and Full-List Eviction

ExpeL (arXiv 2308.10144) trains nothing. It lets an agent run many successful and failed trajectories on
training tasks, compares them pairwise, and asks an LLM to emit **operations on rules** (add, agree,
rewrite, remove). Those operations are then folded into a rule table that prunes itself: rules that keep
getting agreed with climb up, rules that keep getting denied fall to 0 and get dropped.
`agent/expel.py` in `LeapLabTHU/ExpeL` is that book; this problem turns it into a pure function.

## Interface

```python
def solve(rounds, initial_rules=None, max_num_rules=10) -> dict
```

- `rounds`: the LLM critique outputs, one raw string per round.
- `initial_rules`: starting rules as `[(text, count)]`, empty by default.
- `max_num_rules`: used **only** to decide whether the list counts as full; it never truncates.

Returns `rules` (list of `[text, count]`, pruned and sorted), `prompt` (numbered rendering),
`log` (per round: parsed ops, applied ops, the full-list flag, a snapshot), and `count`.

## Input / Output

```
Input:  rounds=["ADD: Check the drawer.\nADD: Wipe the shelf.",
                "EDIT 3: Wipe the shelf first."]
        max_num_rules=1
Output: log[0].parsed  = [["ADD","Check the drawer."], ["ADD","Wipe the shelf."]]
        log[0].rules   = [["Check the drawer.",2], ["Wipe the shelf.",2]]
        log[1].parsed  = [["EDIT 3","Wipe the shelf first."]]
        log[1].applied = []                    # number 3 is out of range (only 2 rules)
        rules          = [["Check the drawer.",2], ["Wipe the shelf.",2]]
        prompt         = "1. Check the drawer.\n2. Wipe the shelf."
```

## Rules

- **Parsing**: each line is matched as `(REMOVE|EDIT|ADD|AGREE)( number)?: [label: ]body`. The verb may
  carry one number (`EDIT 3:`); a number after `ADD` is **dropped**; a `label: ` prefix (a run of letters,
  spaces and digits followed by a colon and a space) is swallowed. The match is **not anchored**, so
  `PADD:` also produces an `ADD`.
- **Three gates**: after stripping, the body must be non-empty, must **end with a period**, and must
  **not contain** any of `ADD`, `AGREE`, `EDIT` as a substring (`REMOVE` is allowed). Two operations on
  one line are both lost — the body column is a greedy `.*` that eats the rest of the line, which then
  hits the banned-word gate.
- **One repair pass** (no accounting yet, only rewriting the operations):
  - `ADD`: dropped entirely if the new body **contains** the full text of an existing rule (the old text
    is a substring of the new one).
  - `EDIT`: if the new body contains an existing rule, it becomes `AGREE <first match index>` carrying the
    **old text** (i.e. the old rule gets +1 and the new text is discarded); with no number, or a number
    larger than the current size, the whole operation is dropped.
  - `REMOVE` / `AGREE`: dropped when the body matches no existing rule.
- **Fixed application order** `REMOVE → AGREE → EDIT → ADD`: REMOVE subtracts 1 (or **3 at once** when the
  list is full), AGREE adds 1, EDIT replaces the text at `number - 1` and adds 1, ADD appends with an
  **initial count of 2**.
- **Full-list test**: `len(rules) >= max_num_rules + 5`, measured **before** the batch starts; rules added
  in a batch can only fill the list from the next batch on.
- **Finalization**: drop rules whose count is `<= 0`, then sort by count **descending and stably** (ties
  keep their previous relative order).
- **Rendering**: `"<index>. <text>"` joined by newlines, indices from 1, no trailing newline; an empty
  ledger renders as the empty string.
- **Validation**: a non-sequence `rounds` or one holding non-strings, a non-integer or negative
  `max_num_rules` (`bool` rejected), or an `initial_rules` entry that is not a `(text, integer)` pair all
  raise `ValueError`.

## Hints

- Deduplication means "old text is a substring of new text"; the reverse direction is not a duplicate. A
  shorter old rule swallowed by a longer new one collides, while a longer old rule does not block a
  shorter new one.
- Two identical `ADD` operations in the same batch both land (deduplication only compares against the
  pre-batch ledger), so the table ends up with two rules of the same text, both at count 2.
- `max_num_rules` never truncates anything: it only affects how hard REMOVE hits. Assuming it caps the
  list is the most common misreading.
- The order is not swappable: AGREE locates by **text**, EDIT replaces by **number**. Run EDIT first and
  the later AGREE can no longer find its original text.
