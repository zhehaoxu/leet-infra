# Solution Notes

## Approach

The deterministic core of the official soochan-lee/RoT repo:

1. **data/problem.py**: `Problem.solve(paradigm='rot')` (L104-135) — inline subquestion Q&A injection + tail-recursion semantics; `_add_thought_xyl` / `_add_answer_xyl` / `_init_question_xyl` (L137-165) — the three-stream (x/y/label) alignment ledger; `ProbGraph.extend` (L310-319) — DFS pre-order deduped expansion of the problem graph; `_flatten_thought` (L369-380) — cot-paradigm pre-order flattening;
2. **data/arithmetic.py**: `Compare.thought` (L48-68), `Add.thought` (L114-133), `Mul.thought` (L248-274) — three thought recursions (carry chains, mul decomposition + tail addition, first/rest-digit comparison);
3. **data/tokenizer.py** (L1-22): the escape/special-token state machine.

Source-verification correction: the todos candidate described "GOAL/ANSWER context-stack ledger (push/pop, stack-depth cap, multi-context injection)" — that is the **paper's inference framework**; the official repo actually implements **seq2seq training-sequence compilation** — subquestions are not pushed on a stack but compiled inline, Q&A with `<THINK>`/`<PAD>`, into a single sequence; the counterpart of the "stack" is `ProbGraph`'s recursive expansion.

Out of judging scope: the torchtext vocab (`build_vocab_from_iterator`), `pad_sequence`/DataLoader, the LSTM/Transformer models and training, `generate()` random sampling (operands are explicit inputs), the ammo/magazine sampling scheduler of `ProblemSet.Iterator`, gpt_eval and the episodes yaml.

Boundary with existing problems:
- **775/776/777**: demo assembly, sample selection, candidate verification; this problem is **recursive problem compilation** — the CoT is not sampled text but deterministically generated from the problem's syntactic structure;
- **rs-toT / rs-lats**: tree search expands at inference time; this problem lives on the training-data-construction side, and subquestions derive mechanically from carry/decomposition rules;
- **data-tokenization collection**: tokenizer problems cover BPE/vocabularies; here the tokenizer is a 22-line state machine (escape + special tokens) — the core is the sequence-alignment ledger.

## Key implementation points (official semantics, pinned item by item)

- **Three-stream length invariant**: `len(x)==len(y)==len(label)` always holds — initially y is one shorter than x; each thought appends `'<THINK>'` to y only (x gets just t_q); a non-tail sub-answer gives x len(t_a) tokens and y len(t_a)-1 `<PAD>`s (refilling 1); the main answer gives x len-1 and y len (another refill); a tail sequence has no main answer and ends with `y[-1]=='<THINK>'`. This invariant is the backbone of the label_invariants case.
- **Tail-recursion semantics** (`T(Add, (a2*10, a1), 'tail')`): `t_q[0]` becomes `'<TAIL>'` (official direct index replacement, assuming the question starts with `<GO>`); t_a=None → no sub-answer emitted; `tail_recursion=True` → the main answer is not emitted either; the official `assert not tail_recursion` blocks "non-tail after tail"; this problem implements it as an equivalent ValueError (message pinned to 'Tail thought is not at the end').
- **Compare's rest-digit trap**: `int(str(l)[1:])` yields 5 for 105 (`'05'` → int drops the leading zero) — the second thought for (105,105) is `(5,5)`; different digit counts mean zero thoughts (the official docstring example at L14, `73847<VS>7243`, showing recursion across differing digit counts is a stale comment — the code's `if digit_l == digit_r` blocks it).
- **Add's carry chain**: on `l_last + r_last >= 10` append `(l_rest, 1)` and **increment after appending** (`l_rest += 1`) — for (9,91) with l_rest=0 the carry thought `(0,1)` still occurs (appended first), then `(1,9)`; `(999,1)` chains `(9,1)→(99,1)` (r_rest=0 blocks a third).
- **Mul's double base and tail arguments**: `l<=1 or r<=1` (memorized ×0/×1) precedes `l<=9 and r<=9` (memorized times table); the tail thought's arguments are `(a2*10, a1)` — **a2 is the high part**; `right=10` takes the `>=10` branch and the `r%10=0` partial product is 0 (pinned by the (12,10) case).
- **Tokenizer state machine**: `'\\c'` swallows the backslash and splits the next character into its own token (`'\\<x>'`→`['<','x','>']`, `'\\\\<x>'`→`['\\','<x>']`); `'<...>'` closes at `>` (including the empty `<>`); a token buffer left in SPECIAL state at loop end is **dropped entirely**.
- **ProbGraph dedup is pruning**: `if prob in self: continue` — duplicate nodes (the two `Mul(11,1)`s of Mul(11,11)) have their subtree expanded only at first occurrence; the returned order is dict insertion order = DFS pre-order.

## Test design (11 cases)

- `tokenize`: special tokens / escape / unterminated / empty `<>` / the `'\\<x>'` and `'\\\\<x>'` double boundary; all expected values pinned by running the official repo;
- `compare_paths`: base / same-digit recursion (105 leading-zero trap) / differing digits zero thoughts / first-digit decides with a single thought — full x/y/label assertions;
- `add_no_carry`: (31,28) full sequence + (5,123) with l_rest=0 blocking;
- `add_carry`: (39,28) carry chain, (999,1) deep chain, (9,91)'s `(0,1)` carry thought — full sequences;
- `mul_tail`: (12,3)/(123,45)/(11,11 duplicate thoughts not deduped) — `<TAIL>` replaced on both streams, no A label, y ending in `<THINK>`;
- `mul_base`: (3,4)/(37284,1)/(9,9 times-table boundary)/(12,10's r=10 decomposition branch);
- `flatten_cot`: pre-order flattening + tail emits only the question + no dedup (both copies for 11,11);
- `graph_order`: 15-node DFS pre-order / multiple roots in order / duplicate roots deduped / subtree dedup / default single root;
- `label_invariants`: 9 groups of (n_thoughts, n_pads, has_tail, n_ans) account checks — equal stream lengths, `x.count('<THINK>')==0`, `y.count('<PAD>')`, tail has no A and ends with `'='`/`'<THINK>'`;
- `purity`; `args_validation`: seven shapes/types for mode/problem/args/roots/text + bool exclusion + the legal tuple-root path of graph.

## Mutation close-out

41 designed (including one two-segment BFS mutation) → **38 CAUGHT + 3 equivalence removals**:

- m-cmp-base-le9: `left < 10` ⟺ `left <= 9` (identical over ints, guaranteed by `_check_args`);
- m-args-len: after removing the shape check, `(1,)`/`(1,2,3)`/`"12"` still raise ValueError at `_question` unpacking or the per-character int check;
- m-val-rot-req: after removing the mode='rot' missing-argument raise, None falls into `_check_problem` and still raises ValueError (different message, same type; m-graph-default-raise is isomorphic and not listed separately).

Coverage: five tokenizer states (escape dropped/sticky, special token missing `>` echo/missing `<` start/unterminated kept), six alignment states (y not shifted/`<THINK>` dropped/T count/`<PAD>` on y vs label/answer x-side untrimmed/label one short/y trimmed), three tail states (token not replaced/sub-answer emitted/main answer emitted), four Compare states (base boundary/digit guard/rest condition/comparison operator), five Add states (carry boundary/increment lost/both guards removed/base half-removed), five Mul states (×1 base/times-table boundary/decomposition boundary/tail arguments swapped/decomposition order swapped), three flatten states (pre-order inverted/tail emits answer/tail token not replaced), four graph states (dedup removed → infinite recursion/DFS→BFS/recursion cut/top-level shape check removed), six validation states.

## Engineering remarks

- **Official docstrings can lie**: Compare's docstring example `73847<VS>7243` shows recursion across differing digit counts, but the code's `if digit_l == digit_r` does not recurse — always trust the code; 7243 in the docstring is likely a typo for 73413.
- **Run the official code before writing expectations**: every fixture's x/y/label was dumped directly from the official repo (after stubbing torchtext, /tmp/rotcheck), avoiding hand-written 32-element label lists; the `(0,1)` thought of (9,91) and the zero partial product of (12,10) were both confirmed only after dumping.
- **Multi-segment mutation replacement**: the DFS→BFS mutation must change both the `def extend` header and the subtree recursion body — two replaces combined into one mutation (the PAIR_MUTS mechanism); a single-segment replacement would leave a syntax error and get misjudged as fatal.
- **Judging protocol**: the judge injects a single `solve`; the four entry points dispatch on the `mode` kwarg (tokenize/rot/flatten/graph).
