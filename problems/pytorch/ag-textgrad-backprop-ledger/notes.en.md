# notes (ag-textgrad-backprop-ledger)

## Sources

- Paper: TextGrad (arXiv 2406.07496). §2 introduces the framework (variables / operations /
  textual gradients / "sum" as a merge of several feedbacks), and Fig 2 draws the `backward()`
  flow. The paper only gives concepts for **traversal order, reduction grouping, and how several
  feedbacks are summed**; it states no executable rules.
- Code: `zou-group/textgrad` main (the paper's official implementation).
  - `textgrad/variable.py`: 11-80 construction (`predecessors = set(...)`, the requires_grad
    check, value type assertion), 151-190 `backward()` (post-order topology + reverse processing +
    clearing only the root), 282-322 `_check_and_reduce_gradients`, 325-367 `_backward_idempotent`
    (the backward of `__add__`).
  - `textgrad/autograd/algebra.py`: 34-101 `Sum`, 104-157 `Aggregate`.
  - `textgrad/autograd/llm_ops.py`: 36-98 `LLMCall.forward` / `backward` base-vs-chain dispatch,
    100-160 `_backward_through_llm_chain`.
  - `textgrad/autograd/reduce_prompts.py`: 1-17 `REDUCE_MEAN_SYSTEM_PROMPT` and
    `construct_reduce_prompt` (one `<FEEDBACK>` per gradient, joined by newlines).
- Only graph structure and bookkeeping are ported: LLM generations become the `grads` table and
  summarisation outputs become the `reduces` table.

## Paper semantics

1. **Feedback is the gradient**: each gradient is a piece of natural language, not a vector, so
   "sum" is concatenation and "mean" is asking another LLM to summarise.
2. **The chain rule lives in a template**: an upstream node's gradient text is dropped into a
   fixed template before the LLM evaluates the downstream variable. This problem pins that
   template **verbatim** (`Here is the combined feedback we got for this specific ... and other
   variables: ... .`) because it determines the gradient text every intermediate node receives.
3. **Variables and operations are separate**: a `Variable` only holds value, role, predecessors
   and gradients; an operation builds a node at call time and attaches a `grad_fn`. The "graph" is
   therefore a by-product of forward calls; there is no explicit graph object.

## Traps in the official implementation

1. **Only the root's gradients are cleared**: `backward()` does `self.gradients = set()` for the
   root alone, so other nodes keep leftovers from a previous round. Calling `backward()` twice
   doubles the intermediate gradient counts and can re-trigger reduction.
2. **Reverse post-order**: `build_topo` recurses into predecessors before appending, and
   processing walks `reversed(topo)`. Sibling branches are therefore **declared first and
   processed last**, which also reverses the order in which a shared input receives gradients.
3. **Idempotent broadcast pushes twice**: `Sum.backward` / `_backward_idempotent` `add` two
   identical gradients per predecessor (the second without `_reduce_meta`). Consequently:
   - a predecessor has `grad_count == 2`, and `get_gradient_text()` repeats the same text twice;
   - two empty strings join into `"\n"` rather than `""`, so **downstream mistakenly believes it
     has a gradient and takes the chain path** — the most hidden chained bug here, invisible to
     ordinary cases.
4. **aggregate pushes to the last predecessor only**: the `for` loop in `Aggregate.backward` only
   assigns, and the push happens after the loop with the final `variable`. So
   `aggregate([a, b])` always gives its gradient to `b`.
5. **aggregate tags itself**: `forward` sets `aggregated_variable._reduce_meta = [...]`, so once
   that node receives two or more gradients it is reduced by **its own** tag; those gradients
   usually carry none, so grouping is empty — no summarisation call, gradients wiped, and an
   empty string is pushed downstream.
6. **Three gates before reduction**: no tag / empty gradient text / exactly one gradient — any of
   them returns as-is. Only past all three does grouping run, and **untagged gradients are
   dropped** there (`new_gradients` only holds per-group results).
7. **`id(variables)` as the group id**: the official code tags reduction groups with the list
   object's `id`, so reusing one list object across two `aggregate` calls merges the groups. This
   problem expresses that explicitly with the `group` field.
8. **The `llm` base/chain dispatch asks whether the response itself has gradients**: if not it
   takes base (let the LLM evaluate the variable directly), otherwise the response gradient is
   carried as chain feedback. Both paths push **one** gradient per predecessor and both skip
   `requires_grad=False` predecessors (which is what system prompts usually are).
9. **Construction-time failure**: a `requires_grad=False` variable may not have predecessors that
   require grad; the official code raises in `__init__`, before any backward pass.

## Normalisations in this problem

The only undecidable points in the official code come from three `set`s (`predecessors`,
`gradients`, the role collection). This problem normalises them to "dedupe in insertion order"
and spells out what that affects:

- `order`: the iteration order of `predecessors` drives the DFS and the reverse post-order;
- `roles`: the order of predecessor roles inside the `sum` / `aggregate` role text;
- `reduce_prompts`: the order in which several gradients of one group are concatenated.

Nothing else deviates from the official code. The `group` field makes `id(list)` explicit: it
defaults to the node id, and giving two `aggregate` nodes the same `group` reproduces the
"shared list object" grouping merge.

## Mutations

43 mutations: **41 substantive ones CAUGHT**, 2 judged equivalent:

- `m-grad-role` (a gradient entry's `role` changed from `feedback to X` to `X`) and
  `m-reduce-role-first` (the reduced role takes the group's first entry instead of the deduped
  join): both only touch the `role` field of gradient entries, while the ledger exposes only
  **values**, counts, prompts and node roles. That `role` neither feeds `grad_text` nor is read by
  any backward pass (backward passes read the **node** role), so external behaviour is identical.
  600 random graphs (549 of which built and backpropagated successfully) produced 0 ledger
  differences for both.
- `m-reduce-gate-text` (dropping the "empty gradient text, skip reduction" gate) SURVIVED the
  first round: with `len > 1`, `"\n".join` always yields at least one newline, so the gate only
  matters for a node that is tagged but holds **zero** gradients. It was caught after adding
  `check_bare_aggregate_root` (an aggregate used as root with no gradients of its own).

## Duplicate check

- No TextGrad / textual-gradient / autograd problem exists in the repo. `agent-react-loop` and
  `agent-context-compaction` cover single-role ReAct and context compression, while
  `ag-metagpt-msgpool-ledger` covers cross-role delivery; none has a "computation graph +
  gradient accumulation/reduction" axis.
- Boundary against `rs-*` (reasoning-side scoring, voting, PRM merging): those scores are
  scalars, whereas these "gradients" are text and the core here is graph traversal and
  accumulation semantics.
