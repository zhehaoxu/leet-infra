# Cumulative Reasoning: Verified Cumulative-State Ledger (Game24)

## Problem

Cumulative Reasoning (arXiv:2308.04371, official repo `iiis-ai/cumulative-reasoning`) organizes its Game of 24 implementation (`Game24/game24-openai.py solve()` L537-590) as a **state-graph growth loop** over three LLM call types — proposer, validity judgement, and verifier: each round samples a reached state, the proposer emits one arithmetic step and the remaining-numbers string, and the verifier returns sure / likely / impossible. Only `likely` grows the graph (with parent pointers, the step, and forbidden entries); `sure` reconstructs the full solution path along parent links. This task makes all three LLM outputs explicit inputs (a `rounds` script) and tests the pure ledger:

```text
mode="solve":  puzzle + rounds + trycnt -> (success, marker, steps)
mode="state":  same inputs -> {"records", "parent", "last_step", "forbidden"}
mode="dist":   index + n_states + trycnt -> sampling distribution list[float]
```

## Ledger rules (pinned to the official code, item by item)

Each round (index i; **any exception silently skips the round** — the official try/except continue wraps the whole loop body):

```text
tmp = rounds[i].chosen                 # sampled reached state (explicit input)
next_step = next.strip()
a = remaining.strip().split('\n')[0].strip()      # first line only
gate A: re.search(r"[^0-9+\-*/.(),=\s]", next_step) -> skip
gate B: next_step or remaining.strip() has no \S -> skip
judgement == 'Invalid' -> skip          # only 'Invalid' blocks; 'Valid' etc. pass
verify == 'sure':
    steps = [equation.strip().split('\n')[0].strip(),
             next_step + ' (left {a})']
    t = tmp; while t != puzzle: steps.append(last_step[t] + ' (left {t})'); t = parent[t]
    steps.reverse(); return (True, i, steps)
verify == 'likely' and a not in records:
    forbidden[tmp].append(next_step); forbidden[a] = []
    records.append(a); parent[a] = tmp; last_step[a] = next_step
verify == 'likely' but a already in records: the whole round is silently dropped (forbidden not appended)
other verdicts (impossible etc.): no action
```

Without success the result is `(False, trycnt, [])` — the marker is **trycnt itself**, not the number of rounds run; rounds beyond trycnt are ignored (the official `for i in range(args.trycnt)`).

The sampling distribution (the official `p` construction, L545-553):

```text
i < 1/2 * trycnt (float division; with odd trycnt the midpoint rounds down):
    n > 1: fill everything with 0.5/(n-1), then overwrite p[0] = 0.5 (root bias, encouraging restarts)
    n == 1: p = [1.0]
otherwise: uniform 1/n
```

## Traps

- **Duplicate states drop the whole round**: likely with `a` already in records does nothing — `forbidden[chosen]` is not appended and f/last_step are not overwritten (the dup_drop case pins forbidden holding a single entry).
- **Gate B's second operand is the full stripped remaining**, while `a` is the first line — for `'6, 8  \njunk'` the value of `a` is `'6, 8'` (the first line is stripped again at the tail), and junk never enters the state.
- **last_step stores the full stripped next** (internal newlines included), and the reconstruction keeps the newline when appending `' (left {t})'`.
- **Structure of sure's steps**: after the reverse, the last item is the equation's first line (the merged expression) and the second-to-last is this round's next — the equation is the expand LLM's output placeholder.
- **The official success path returns the expand LLM's merged output**; this task has no expand LLM and returns the deterministic artifact directly (the `'\n'.join(tmp_steps)` payload fed to expand) — the deviation is recorded in the notes.
- **Silent-skip semantics**: a chosen not in records (impossible via numpy.choice officially, but explicit here) raises `forbidden[tmp]` KeyError, which the officially co-located except swallows — the round is skipped, not an error.
- The first-half test uses **float division** `i < 1/2 * trycnt`: with trycnt=5, i=2 is still biased and only i=3 is uniform.

## Judge environment

No numpy / guidance / openai dependency — sampling (choice), the retry wrapper (run's MAX_RETRY), the expand merge, and the sympy check are all out of scope; the p vector is built with pure python floats (same values as numpy's fill).
