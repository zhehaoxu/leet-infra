# CoCoGen: Code-to-Graph Bidirectional Parsing Ledger (Graph-String Parsing + add_edge Code Emission + Code Recovery)

## Problem

CoCoGen (arXiv:2210.07128, official repo `reasoning-machines/CoCoGen`, migrated from `madaan/cocogen`) represents commonsense graphs as python code (`class ExplanationDAG` + `add_edge` lines); `src/converters/explagraphs/relation.py` provides deterministic bidirectional conversion. Source correction: the original todos description ("cycle detection, root/leaf constraints, edge-label validity") does not hold — the repo has **no** standalone graph validator; the deterministic core is the three-way ledger (graph-string parsing / code generation / code recovery) plus the GraphAlgos root ledger and the three case-conversion functions. All pieces are explicit inputs; the task tests the pure ledger:

```text
mode="parse":    graph_str -> {"nodes", "from_to", "from_to_typed"}
mode="to_python": belief + argument + stance + graph_str (+prompt_part_only) -> code string
mode="to_graph": code_str -> {"belief", "argument", "stance", "graph"}
mode="roots":    src_dest adjacency -> in-degree-0 node list
mode="case":     text + case("camel"|"snake"|"desnake") -> converted text
```

## Ledger rules (pinned to the official code)

`mode="parse"` (official `get_nodes` L81-113):

```text
graph_str[1:-1] strips ONE character from each end (regardless of being parentheses)
split on ')('; segments without ';' are skipped entirely
segments containing ';' must split on ';' into exactly 3 parts (otherwise unpack ValueError); each part stripped
nodes recorded in first-appearance order (from before to)
from_to / from_to_typed initialized only on the from's first appearance
when to is already in from_to_dict[from], NEITHER ledger appends
  (a second from->to edge — even with a different relation — is dropped entirely)
```

`mode="to_python"` (official `graph_to_python` L10-79, relation variant): fixed skeleton
`class ExplanationDAG:` + blank line + `__init__` + `belief =`/`argument =`/`stance =`
(three lines, **interpolated verbatim without escaping**) + blank line +
`# create a DAG to {stance} belief using argument`; `prompt_part_only=True` returns here.
Otherwise parse the graph string, write the root nodes (in-degree 0, node-addition order)
as a `begin = ["root1", "root2"]` line (**quoted**), then emit
`add_edge("from", "rel", "to")` per edge in `from_to_typed` dict order; one trailing blank line.

`mode="to_graph"` (official `python_to_graph` L118-178, relation variant): regex
`belief = "(.*?)"` etc. extract the three fields (missing -> AttributeError propagates);
per line containing `add_edge`: `split("(")[1]` takes everything after the **first** opening
parenthesis (a `(` inside the target drops everything after it), `[:-1]` chops the last
character, then split on `", "`+quote (comma+space+quote) into exactly 3 parts
(out-of-range IndexError propagates), strip each, join as `(source; rel; target)`.
After all lines, unified cleanup: delete all `"`, `;` -> `; `, `"  "` -> `" "` (**single pass**,
three spaces collapse to only two).

`mode="roots"` (official `GraphAlgos.root_nodes` over `to_nx_graph`): in-degree 0, in node
addition order (`add_edge(src, dest)` adds src before dest); **a src with empty dests never
enters the graph**; a pure cycle has no roots and returns `[]`.

`mode="case"` (official `src/converters/utils.py`): camel = strip a trailing period +
`split(' ')` + join per-token `capitalize()` (**lowercasing the rest**); snake = strip a
trailing period + `re.sub(r'[\s\W]', '_', ...)` (each space and each non-alphanumeric char
becomes ONE underscore) + `lower()` + `strip()` (whitespace only; a trailing `_` produced by
a stripped period survives); desnake = `split('_')` then `' '.join`. Empty strings raise
IndexError at camel/snake's `title[-1]` (uncatched officially).

## Key quirks (official behavior)

- parse strips **characters**, not parentheses: `X(a; rel; b)Y` yields nodes `(a` and `b)`;
- a second from->to edge with a different relation is **dropped entirely** (the dedup check gates both ledgers);
- to_python's belief/argument/stance are not escaped (a belief containing `"` produces broken code — faithful);
- to_graph's line parsing relies on `split("(")[1]` + `[:-1]`: content after a second `(` is lost, and a line without a trailing `)` loses its last character; a double space breaks the `", "` separator -> immediate IndexError;
- the cleanup triple is quote-delete -> semicolon-plus-space -> double-space fold, with **order and single-pass** semantics (`';  '` stays, three spaces do not fully collapse);
- snake's strip only removes whitespace: `Do A (Thing)!` -> `do_a__thing__` (trailing `_` kept).

## Input conventions

`graph_str` is a `(from; rel; to)(...)` string; `src_dest` is a `{from: [to...]}` adjacency
map; snake/desnake are not inverses (underscores are ambiguous).
