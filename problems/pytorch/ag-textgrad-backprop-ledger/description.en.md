# TextGrad Textual Backprop Graph Ledger

TextGrad (arXiv 2406.07496) treats natural-language feedback as a gradient: a `Variable`
is a node in a computation graph, every operator call builds a new node carrying a
`grad_fn`, and `root.backward()` pushes feedback back through the graph. This problem peels
that autograd down to a pure function — LLM generations and LLM evaluations become explicit
arguments, leaving only **graph building, traversal, gradient accumulation and reduction**.

In the official implementation (`zou-group/textgrad` main) every container is a `set`:
`predecessors`, `gradients` and the role-description collection have no guaranteed iteration
order. This problem normalises all three to "dedupe in insertion order"; every other rule
follows the official code line by line.

Implement `solve(nodes, grads, reduces=None, root=None, rounds=1)`.

## Graph and nodes

`nodes` is given in **build order**, with unique `id`s:

```
{"id": "q", "kind": "input", "value": "3 * 7", "role": "question",
 "requires_grad": True, "inputs": [...], "system": "sys", "group": "G"}
```

- `kind = "input"`: a leaf; `value` / `role` are required, `requires_grad` defaults to `True`.
- `kind = "llm"`: one LLM call; `inputs` lists predecessor ids and the optional `system` points
  at a system-prompt node; `value` is this call's generation.
- `kind = "sum"` / `"aggregate"`: concatenate the predecessors' texts; `inputs` may be empty.
- `kind = "add"`: `Variable.__add__`, taking **exactly two** operands.
- Only `aggregate` uses `group` (defaults to its own id); other kinds ignore it.

## Forward graph building

- Values: `input` / `llm` take the declared `value`; `sum` / `add` / `aggregate` join their
  predecessors' values with a newline `"\n"`.
- The three role texts differ (they are embedded verbatim into the backward feedback template,
  so they must be exact):
  - `sum`: `a combination of the following: <deduped roles, comma+space>`
  - `add`: `<left role> and <right role>` (**no** dedupe)
  - `aggregate`: `a combination of the following variables: <deduped roles>.` (trailing period)
- `requires_grad`: `sum` / `aggregate` use "any predecessor requires grad", `add` ORs the two.
- An `llm` node's predecessor order is `[system] + inputs` — the system prompt comes first.
- `aggregate` attaches its reduction tag `{"op": "mean", "id": group}` to **itself**.
- Hard constraints: `requires_grad=False` while a predecessor requires grad, an `add` with a
  number of operands other than 2, a duplicate `id`, an unknown predecessor id, an unknown
  `root`, an empty graph, and `rounds < 1` all raise `ValueError`.

## Backward order

Starting from `root` (default: the last declared node), run a **post-order** DFS over the
predecessors in declaration order, then process the nodes in **reverse**.

- Only the **root** has its gradients cleared; intermediate nodes keep whatever was left from a
  previous round (with `rounds > 1` that keeps accumulating).
- A node with `requires_grad=False` is skipped entirely: gradients untouched, `grad_fn` not
  called, one entry in `skipped`.
- Every other node first "checks and reduces its gradients", then calls its `grad_fn`.
- Topology is rebuilt each round; `order` concatenates the per-round processing orders.

Reverse post-order means sibling branches are **declared first, processed last**, which directly
changes the order in which a shared input receives gradients.

## The three backward passes

- `sum` / `add` (idempotent broadcast): let `T` be this node's gradient text. For **every**
  predecessor push
  - the empty string when `T == ""`, otherwise
    `Here is the combined feedback we got for this specific <predecessor role> and other variables: <T>.`
  - and push it **twice** per predecessor (the official code calls `add` twice); only the
    **first** copy carries this node's reduction tag, which is also appended to the predecessor
    node itself. The second copy carries nothing.
- `aggregate`: only the **last declared** predecessor receives one gradient, carrying the tag
  (the official `for` loop only assigns, and the push happens after the loop, so exactly one
  predecessor ever receives a gradient).
- `llm`: if its own gradient text is empty it takes the `base` path, otherwise `chain`; a
  predecessor with `requires_grad=False` is skipped; each predecessor receives exactly one
  gradient, read from `grads["<node>:<base|chain>:<predecessor>"]`.

## Gradient reduction

While processing a node, if it carries a reduction tag, its gradient text is non-empty, and it
holds **more than one** gradient, reduction runs:

1. group gradients by the tag id attached to **each gradient**, groups in first-appearance order;
2. one summarisation call per group; the prompt wraps every gradient in the group as
   `<FEEDBACK>value</FEEDBACK>` and joins them with newlines;
3. the reduced value comes from `reduces[group]`, its role is the deduped roles joined by commas,
   and the reduced gradient carries **no** tag;
4. untagged gradients are dropped here; when no group can be formed the result is an **empty
   list** and no summarisation call is made at all.

## Output

```
{
  "order": ["a", "q"],
  "skipped": [],
  "steps": [{"round": 1, "id": "a", "skipped": false, "in": [], "out": [],
             "reduced": false, "called": true, "path": "base",
             "pushed": [{"to": "q", "count": 1}]}],
  "grad_text": {"q": "...", "a": ""},
  "grad_count": {"q": 1, "a": 0},
  "values": {"q": "3 * 7", "a": "20"},
  "roles": {"q": "question", "a": "answer"},
  "reduce_prompts": ["<FEEDBACK>...</FEEDBACK>"],
  "rounds": 1
}
```

- `steps` follows the processing order: `in` is the gradient values before reduction, `out`
  after it, `reduced` says whether the reduction branch ran, `called` whether `grad_fn` ran, and
  `path` exists only on `llm` nodes.
- `grad_text` is each node's final gradient text (multiple entries joined by newlines);
  `grad_count` is how many entries there are.
- `reduce_prompts` holds the summarisation prompts **actually issued**, in call order.

## Example

```
nodes = [
  {"id": "q", "kind": "input", "value": "3 * 7", "role": "question"},
  {"id": "a", "kind": "llm", "inputs": ["q"], "value": "20", "role": "answer"},
]
grads = {"a:base:q": "the question asks for 21"}
led = solve(nodes, grads)
led["order"]        # ["a", "q"]
led["grad_text"]    # {"q": "the question asks for 21", "a": ""}
led["grad_count"]   # {"q": 1, "a": 0}
```

## Constraints

- Pure function: no network, no file I/O, no mutation of the arguments; identical inputs must
  produce an identical ledger.
- Reduction results and LLM gradients may only come from the argument tables; a missing key
  raises `KeyError`.
- Graph-building violations raise `ValueError`.
