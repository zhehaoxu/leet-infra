# notes — ag-rewoo-plan-ledger (order 721)

## Source of truth

- Official repo billxbf/ReWOO (the paper's own); the ReWOO driver lives in **`algos/PWS.py`** (there is no rewoo.py under algos/; react.py/notool.py are the comparison arms). Three sections: `_parse_plans` / `_parse_planner_evidences` / `_get_worker_evidences` plus the log/cost assembly in `run()`.
- Price table `utils/util.py::get_token_unit_price`: text-davinci-003=0.00002, gpt-3.5-turbo=0.000002, gpt-4=0.00003, LLAMA_WEIGHTS (tloen/alpaca-lora-7b, rewoo/planner_7B)=0.0, unknown → `ValueError("Model not found")`. The tool token price is **fixed at the davinci-003 rate** regardless of planner/solver models. Google costs 0.01 per query.

## Key ledger points (official quirks recorded)

1. **Multi-digit ids fail**: `if len(e) == 3` — `#E12 = Google[q]` becomes `No evidence found` and the call is discarded; the key still enters planner_evidences, so later `#E\d+` references substitute `[No evidence found]` (the literal string passes through into worker_evidences).
2. **Evidence line parsing**: `line[1]=='E' and line[2].isdigit()` (`#e1`/`#-1` unrecognized); `split("=", 1)` with both sides stripped; **duplicate evidence lines: later wins** (dict overwrite, first-insertion position kept).
3. **split("[",1) + [:-1] blind trim**: `Google[foo][bar` → tool_input `foo][ba`; the RHS of `=` is stripped first, so a "leading-space tool name" quirk cannot occur, but `Google [x]` (space before bracket) → tool=`Google ` → unknown tool.
4. **Substitution semantics**: `for var in re.findall(r"#E\d+", tool_input)` (original order, duplicates included) → `if var in worker_evidences` (**the worker evidence table**, not planner) → `tool_input.replace(var, "[" + ev + "]")` **replaces all occurrences** — placeholders appearing inside inserted evidence are hit by later vars (`#E1=Quote #E2 line`, `#E3=LLM[#E1 and #E2]` → `[Quote [cats] line] and [cats]`); forward references stay literal and still count into LLM tokens.
5. **worker_log pairs by index**: `plans[i]` ↔ `#E{i+1}` (not by evidence existence); surplus evidence is computed but excluded; a missing pair crashes (official KeyError, this problem raises ValueError — divergence recorded).
6. **Token accounting**: LLM `len(tool_input + evidence) // 4` (floor, on the **substituted** call); Google counts calls; Calculator's token accounting depends on langchain's `LLMMathChain` prompt template constant — **excluded** (a Calculator entry in tools behaves as a plain tool; divergence recorded).
7. **steps = len(plans) + 1** (+1 for the Solver call).

## Not modeled

- Real LLM/Google/Calculator calls, planner/solver prompt templates and fewshots (planner_log/solver_log reduce to prompt+output concat with no independent information), wall_time, the concrete `WORKER_REGISTRY` implementations (outputs are explicit inputs).

## Authoring lessons

- **The `=` RHS is stripped as a whole first**: the planned "tool name with leading space" case was silently eaten by strip — the quirk needs `Google [x]` (space before the bracket); verify the official strip order character by character before writing anchors.
- **Plans pair with evidence by index**: an early anchor with 3 plan lines but 2 evidences raised ValueError directly — in random cases plan count and `#E{i+1}` existence are independent; verify the ValueError symmetry (solve/oracle failing identically) separately.
- **`len(a+b)//4 → len(a)+len(b)//4` is not an equivalent mutation** (precedence: the division applies only to len(b)); it is genuinely CAUGHT. The true equivalent needs parentheses `(len(a)+len(b))//4` (SURVIVED confirmed and recorded).
- All 22 substantive mutations CAUGHT (log-use-planner first showed PATTERN NOT FOUND due to a wrong quote/paren match surface; unknown-model-free was masked by the "missing worker output" ValueError — both fixed and re-verified CAUGHT).
