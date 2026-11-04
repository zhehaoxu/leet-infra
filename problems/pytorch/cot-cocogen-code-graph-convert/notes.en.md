# Solution Notes

## Approach

The deterministic core of reasoning-machines/CoCoGen (source correction: the original todos description "cycle detection, root/leaf constraints, edge-label validity" does not hold — grep finds no is_valid/validate/cycle checker in the repo, and GraphAlgos has only the five thin functions topo_sort/root_nodes/to_nx_graph/get_predecessors/get_successors):

1. **relation.py ExplagraphPythonConverterRelation.get_nodes** (L81-113): `(from; rel; to)(...)` graph-string parsing ledger — character-level end stripping, `)("` splitting, `;`-less segments skipped, 3-part unpacking, strip, first-appearance dedup, **whole-edge drop on duplicate to**;
2. **graph_to_python** (L10-79, relation variant): fixed skeleton + `begin = [...]` (root ledger) + per-edge `add_edge("from", "rel", "to")` emission;
3. **python_to_graph** (L118-178, relation variant): regex field extraction + per-line `split("(")[1][:-1]` + `", "`+quote splitting + three-part strip + the cleanup triple (quote delete / semicolon-plus-space / double-space fold);
4. **GraphAlgos.root_nodes over to_nx_graph** (utils/algo_utils.py): in-degree 0 in addition order;
5. **src/converters/utils.py three functions**: to_camel_case (trailing period + capitalize join), to_snake_case (trailing period + `[\s\W]`->`_` + lower + strip), from_snake_to_normal_str (`_` split back to spaces).

Out of judging scope: OpenAI calls and prompt assembly (make_codex_prompt etc.), the tree/text/literal converter variants (same mechanisms, relation variant taken), the proscript/propara/boolgen families (process-script/attribute conversion, a different paper task), networkx itself (root ledger self-implemented), compile_code_get_object's exec fallback (syntax-error retry truncation, not this ledger), struct_eval/graph_metrics GED/isomorphism/JSD metrics (graph comparison, no parsing semantics).

Boundaries with existing problems:
- **775-780**: demo assembly / candidate scoring / backward verification / RoT sequence compilation / CR state graph / STaR evaluate-writeback; this problem is a **representation-layer bidirectional conversion ledger** — inputs and outputs are plain text, no control-flow state machine;
- **776's structured parsing**: parsed candidates feed into score comparison; here parsing is the terminus (round-trip identity is the mechanism itself).

## Implementation details (pinned to the official code)

- **parse's end stripping is character-level**: `graph_str[1:-1]` — empty/short strings are safe (slices clamp to empty), `"(a)"` leaves `a` with no `;` and parses to nothing;
- **3-part unpacking**: `from_node, edge_type, to_node = edge.split(";")` — a segment whose `;` count != 2 raises ValueError (uncatched officially); `"(noise)"` has no `;` and is skipped entirely;
- **whole-edge drop on duplicate to**: `if to_node not in from_to_dict[from_node]:` gates both `from_to_dict.append` and `from_to_typed.append` — the r2 edge of `(a; r1; b)(a; r2; b)` enters neither ledger;
- **to_python skeleton line by line**: the class line carries **two** `\n`s, the three fields interpolate verbatim (quotes unescaped), the comment line interpolates stance, roots quoted via `", ".join(f'"{n}"')`, and the result ends with `py_source + "\n"`;
- **root ledger**: `to_nx_graph` only calls `add_edge` — **a src with empty dests never enters the graph** (`{"b": []}` has roots `[]`, not `["b"]`); `add_edge(src, dest)` adding src before dest fixes the traversal order;
- **to_graph line parsing**: `line.split("(")[1]` truncates everything after a `(` in the target (`add_edge("a", "r", "b (x)")` yields `(a; r; b)`); `[:-1]` chops one trailing character — a line without `)` loses its final quote but still parses; the `", "`+quote separator mismatches on double spaces -> fields has 1 part -> `fields[1]` IndexError;
- **cleanup triple order**: delete `"` (residual quotes cleaned globally) -> `;` -> `; ` (adds a space after junk segments' semicolons too) -> `"  "` -> `" "` **single pass** (three spaces fold by one level only);
- **snake's character accounting**: `re.sub(r'[\s\W]', '_', ...)` turns each whitespace/non-alphanumeric char into **one** underscore (no `+` quantifier) — a double space gives a double underscore; `strip()` removes whitespace only, so the trailing `_` produced after stripping a period survives; `capitalize()` uppercases the first letter and **lowercases the rest** (`doA` -> `Doa`, unlike `token[0].upper()+token[1:]`);
- **empty strings**: camel/snake's `title[-1]` raises IndexError on `""` — the validation layer does not block it; the official exception propagates.

## Test design (12 cases)

parse 2 (basic graph + an 8-path quirk case: character-level stripping, `;`-less segments, empty and `(a)`, both duplicate-to drops, 4-part ValueError, from with surrounding whitespace), to_python 3 (docstring example byte-exact, prompt_only truncation, root order + unescaped interpolation), to_graph 2 (round-trip identity + a 7-path quirk case: missing-belief AttributeError, short-line/double-space IndexError, no-`)` line, `(` truncation inside target, junk tail, clean round-trip), roots 1 (double-root order, pure cycle [], empty map, empty-dests src not in graph), case 3 (camel including capitalize's lowercasing and empty-string IndexError, snake four variants, desnake), args_validation 1 (11 invalid-input paths). All expected values were pinned by exec-sandboxing the official source (ast-extracted class/function segments, stub GraphPythonConverter/List/Dict, networkx 3.7 passing through the official GraphAlgos), including a to_python->to_graph round-trip identity assertion.

## Deviations from the official code

- Official `get_nodes` returns the tuple `(nodes, from_to_dict, from_to_dict_typed)` (typed values as tuple lists); this task's parse returns a dict (typed values as tuple lists for comparability).
- Official `graph_to_python`/`python_to_graph` are methods with internal debug prints; prints removed, line-by-line skeleton semantics unchanged.
- The official root ledger goes through `networkx.DiGraph`; this task self-implements the add_edge order + has_parent set (line-checked against nx's node-addition order / in-degree semantics).
- `GraphAlgos.topo_sort` (a one-line nx forward) and predecessors/successors are out of scope (no incremental branching semantics).
