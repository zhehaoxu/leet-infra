# notes (ag-got-operations-ledger)

## Sourcing

- Official repo spcl/graph-of-thoughts (ETH Zurich, BSD); core in `graph_of_thoughts/operations/operations.py` (900 lines) + `thought.py` (117 lines) + `graph_of_operations.py`. README examples live in `examples/` (sorting/set_intersection/keyword_counting/doc_merge).
- Paper arXiv 2308.09687 (Besta et al., "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"). The framework models reasoning as an operation graph over an arbitrary DAG; this problem uses a **linear operation chain** (`append_operation` semantics). Multi-predecessor merging (`add_operation`) is out of scope — the linear chain already covers all single-operation semantics; DAG-topology scheduling (can_be_executed readiness) is left as a future candidate.

## Key official quirks (all verified line-by-line)

1. **Improved products reset** (biggest trap): official `ValidateAndImprove` does `current_thought = Thought({**current_thought.state, **state_update})` after each improvement — a **brand-new Thought** with score reset to 0.0 and scored/valid/validated/solved/compared all reset. Same for `Improve`/`Aggregate`/`Generate` products. Only the `from_thought` path (Score/Selector/KeepBestN/KeepValid/GroundTruth and validate's initial attempt) copies flags. The first solution version carried the score through improvements — caught by anchors and fixed.
2. **Zip truncation**: `Score._execute`'s combined branch does `for thought, score in zip(previous_thoughts, scores)` — when the parser returns fewer scores, the extra thoughts silently vanish (no log, no error). The individual branch calls scoring_function per thought.
3. **Attempt counting**: inside `while True` it validates, appends, then checks `if not improve or valid or current_try >= num_tries: break`, incrementing `current_try` after the improvement. Persistently invalid → exactly num_tries+1 validations (the last validation happens after the final improvement). `current_try` starts at 0.
4. **Aggregate's ordered merge**: `for thought in sorted(previous_thoughts, key=lambda t: t.score): base_state = {**base_state, **thought.state}` — ascending stable; **higher scores merge later** and dict merge favors the later value; unscored thoughts participate with the default 0.0 (the official code never checks scored!). Official comment: "applied in order of score".
5. **KeepBestN's fallback**: `try: sorted(...) except: degrade to keeping only isinstance(i.score, float) thoughts`. The bare `except:` triggers on any TypeError from sorted (e.g. score=None mixed in). **int is not float** (`isinstance(1, float) is False`), so int-scored thoughts are dropped in the fallback — the case `scores=[None, 2.5, 1, 3.5]` with `n=2` pins both the None drop and the int drop.
6. **KeepValid pass-through**: `if not thought.validated or thought.valid` — unvalidated thoughts are kept outright (the official code only logs "has unvalidated thoughts").
7. **GroundTruth swallows exceptions**: `try: new_thought.solved = evaluator(state) except: new_thought.solved = False` — bare except; on exception solved=False and compared_to_ground_truth=True (the setter is unconditional).
8. **Generate idle skip**: `if len(previous_thoughts) == 0 and len(self.predecessors) > 0: return` — thoughts stays [], subsequent operations cascade; overproduction only logs a warning without truncation. With no predecessors, `previous_thoughts = [Thought(state=kwargs)]` (kwargs being the problem-level base_state).
9. **Predecessor assert vs empty thoughts**: the official assert checks `len(self.predecessors) > 0` (graph level), always true past the first operation in a linear chain — **zero thoughts do not error** (Aggregate even has an explicit `if len(previous_thoughts)==0: return`). The first solution version conflated the two with `if not prev: raise`; the random cross-check caught it (a generate's zip truncation produced zero thoughts → aggregate wrongly raised ValueError).

## Authoring engineering

- The judging environment has no official dependencies (graph_of_thoughts pulls langchain/openai) — pure scalarization was unobstructed: all LLM calls are made explicit as inputs (scores/valids/updates/parsed/results sequences in consumption order).
- The linear chain's "parent count" trap: the first generate has no predecessor → single `Thought(kwargs)` parent → extra parsed slots get zip-truncated. Multi-children from a single parent requires putting multiple states inside parsed[0]. The first anchor set assumed "slots = thought count" and all broke under double zip truncation — fixed to single-slot multi-state.
- base_state pollution: the first generate's parent state is base_state, so merged states carry the base keys — anchors calling `_compare(..., {"p": 1})` would need `p` in every expected state; passing None is cleaner.
- Random cross-check symmetry: random chains can trigger IndexError (selector out of range) / TypeError (aggregate sorting with None scores) / ValueError — both implementations behave identically, so only symmetry is required (same exception type or both succeed).
- **20 mutations all CAUGHT + 2 equivalents**: `selector not copying thoughts` (`prev[i]` direct reference vs `_from_thought`) and the `aggregate dict wrapping style` — under official semantics states are shared references and never mutated in place, so both forms are observationally equivalent; noted.
- Red lines: prompter/parser prompt templates and parsing regexes are out of scope (pure string engineering); GraphOfOperations' roots/leaves bookkeeping (add_operation topology) is out of scope — it degenerates to identity under a linear chain.

## Future candidates

- Multi-predecessor merging + DAG readiness scheduling (can_be_executed + the controller's execution loop) could make a Hard problem: the operation-graph ledger (topological readiness + multi-predecessor thought concat order = predecessors list order).
