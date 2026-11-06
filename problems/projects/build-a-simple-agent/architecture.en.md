# Architecture Map: Build a Simple Agent

> This is not a new judging format. It is six stateless problems chained into milestones: **frozen official scaffold + the blank you fill in on this step**. The judge replays official modules only, never your own code from earlier steps.

## In One Sentence

No model ever enters the judging sandbox. Every "what the model said" is an explicit fixture argument; what you write is the pure logic around it: schema validation, loop state machine, memory ledger, retry and breaker, multi-turn acceptance, architecture review.

## Milestone Dependency Graph

```text
                       +-----------------------+
   user turn --------> |  01 tool_schema       |  validate(spec, args)
                       +-----------+-----------+
                                   |
                                   v
                       +-----------------------+
   fixture LLM ------> |  02 react_loop        |  trace / final / why
                       +-----+-----------+-----+
                             |           |
               +-------------+           +-------------+
               v                                       v
    +----------------------+             +----------------------+
    | 03 memory_ledger     |             | 04 retry_breaker     |
    | append / compact     |             | plan / breaker       |
    +----------+-----------+             +-----------+----------+
               |                                     |
               +----------------+--------------------+
                                v
                   +------------------------------+
                   | 05 dialog_runner  (FINAL)    |
                   | turns / calls / answers      |
                   +--------------+---------------+
                                  v
                   +------------------------------+
                   | 06 L4 design review          |
                   +------------------------------+
```

Dependencies describe a **recommended path only**, not a lock: every `starter.py` already bundles all earlier official implementations, so you can start at any step.

## Why Each Step Exists

| Step | Milestone | Why it exists | Verification |
| --- | --- | --- | --- |
| 01 | Tool schema validator | Make "is this call legal" a pure function first, so the loop has an assertable boundary | L1 / L2 |
| 02 | ReAct loop state machine | Give the agent an observable round boundary; even the stop reason must be decidable | L1 / L2 |
| 03 | Memory ledger and deterministic compaction | Write traces into a replayable ledger so long dialogues stay bounded and reproducible | L1 / L2 |
| 04 | Retry backoff and breaker budget | Failure paths are deterministic too: backoff is a computed plan, never a real wait | L1 / L2 |
| 05 | Scripted multi-turn end-to-end acceptance | Assemble the previous four blocks into one data flow and assert call sequences and final answers | L1 / L2 |
| 06 | L4 design review | Close with an ordering/judgement question about "why cut it this way", not finger memory | L4 |

## The Whole Data Flow

```text
user turn -> fixture replies -> schema validate -> ReAct trace
          -> ledger append/compact -> retry plan -> per-turn report
          -> final answer -> L4 stage audit
```

## Judging Contract

- **Stateless**: each step's inputs are only the fixtures in the statement and the official modules under `scaffold/`; your earlier code is never replayed.
- **Deterministic**: no randomness, no network, no sleep, no global state; generated text, probabilities, logprobs and tool results are all explicit arguments.
- **Four-layer stack**: L1 oracle comparison field by field, L2 invariants (idempotence, round-trip, boundedness, reproducibility), L4 design review (deterministically gradable). This project has no training step, so L3 fixed-seed bands do not apply.
- **Separate records**: project steps live under `problems/projects/`, are absent from `collections.json`, do not consume the global `order` pool, and keep progress in `data/project_progress.json`.
