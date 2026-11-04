# RoT: Recursive Solving Sequence Compilation

## Problem

RoT (Recursion of Thought, arXiv:2306.06891, official repo `soochan-lee/RoT`) **compiles** an arithmetic problem under the rot paradigm **into a single seq2seq training sequence**: subquestion Q&A pairs are injected inline into the problem sequence, and a tail-recursive subquestion replaces the main answer. This task asks you to implement four deterministic ledgers exactly per the official semantics (pure strings/lists, no model):

```text
tokenize:   escape-/special-token-aware character splitting
rot:        problem -> (x, y, label) training sequence
flatten:    pre-order flattening of the cot-paradigm thought chain
graph:      ProbGraph problem-graph expansion (DFS pre-order, deduped)
```

For `mode="rot"`, the input is `problem ∈ {'compare','add','mul'}` plus an operand pair `(left, right)`; the output is a triple with `label: PAD=0 / Q=1 / T=2 / A=3`.

## Ledger rules (pinned to the official code, item by item)

```text
x = tokenize(question)              # starts with '<GO>...'
y = x[1:]                           # shifted by one (seq2seq teacher forcing)
label = [Q] * len(y)
for each thought = (sub, sub_args[, 'tail']):
    t_q = tokenize(sub's question)
    non-tail: t_a = tokenize(sub's answer)
        x += t_q;  x += t_a
        y += t_q + ['<THINK>'];  y += ['<PAD>'] * (len(t_a) - 1)
        label += [T] * (len(t_q) + 1);  label += [PAD] * (len(t_a) - 1)
    tail: t_q[0] becomes '<TAIL>' (on both x and y), no sub-answer, and the
        main answer is not emitted; a non-tail thought after a tail one is invalid
non-tail finish:
    x += answer[:-1];  y += answer;  label += [A] * len(answer)
```

The thought recursions of the three problem classes (`args` are non-negative int pairs):

- **Compare** (`<GO>l<VS>r=`): double base case (`l<10 and r<10` emits `<GT>/<LT>/<EQ>` directly); recursion only when digit counts match — compare the first digit (`int(str(l)[0])`), and if equal compare the rest (`int(str(l)[1:])`, **leading zero dropped**: rest of 105 is 5); different digit counts mean zero thoughts.
- **Add** (`<GO>l+r=`): base `l<10 and r<10`; compare last digits (`l%10, r%10`) first; on `>=10` carry, append `(l//10, 1)` and do `l//10 += 1`; append the rest-digit addition only if `l_rest>0 and r_rest>0`.
- **Mul** (`<GO>l*r=`): base `l<=1 or r<=1` (memorized ×0/×1) and `l<=9 and r<=9` (memorized times table); for `r<10` decompose into `(l%10, r)` and `(l//10, r)`, otherwise `(l, r%10)` and `(l, r//10)`; finally append a **tail** thought `(a2*10, a1)` (a1/a2 are the two partial products) — the main answer is that addition's result, computed one level down.

tokenizer: `'\\c'` escape turns the next character into its own token (backslash swallowed); `'<...>'` closes at `>` into one token (including the empty `<>`); an unterminated `'<...'` is dropped entirely.

ProbGraph: `extend` dedupes by node `(problem, args)` (first occurrence expands its subtree recursively) and returns the DFS pre-order expansion; the cot-paradigm `_flatten_thought` flattens pre-order (subquestion → recursive expansion → subanswer; tail emits only the `'<TAIL>'` question, no answer) and **does not dedupe**.

## Traps

- **y receives one more token than x per thought**: each thought appends `'<THINK>'` to y only, while the main answer lets x skip its last token — the invariant `len(x)==len(y)==len(label)` holds via "base: y=x-1 → each thought: +1 → answer: +1"; a tail sequence ends with `y[-1]=='<THINK>'`.
- **`<TAIL>` replaces the subquestion's first token on both x and y**, and neither the main answer nor the sub-answer is emitted (no A in label); but `'<STOP>'` inside non-tail sub-answers stays in the sequence.
- **`int(str(l)[1:])` drops leading zeros** (105 → 5), and **the carry thought `(0,1)` still occurs when `l_rest=0`** (9+91) — both are genuine official behavior.
- Mul decomposition edge: `right=10` takes the `>=10` branch (the `r%10=0` partial product is 0).

## Judge environment

No torch / torchtext dependency — the vocab/`pad_sequence` shell of the official `Problem.solve` is out of scope; only the pure token/label ledger is tested; random sampling (`generate`) is out, operands are explicit inputs.
