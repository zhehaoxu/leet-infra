# Regex compilation artifact: token-level Index construction ledger

Constrained decoding must answer "which tokens are legal from the current state, and where do they lead" in O(1) per step, so outlines-core settles this ledger once at **compile time**: cross the byte-level DFA with the vocabulary and compress it into a `state -> token_id -> next_state` index (the `Index` in `dottxt-ai/outlines-core`, `src/index.rs`). Given a byte-level DFA (state 0 is the anchored start; unwritten bytes fall to the sink), reproduce the full construction algorithm.

## Procedure

1. **Enumeration**: start from state 0, process states with a LIFO stack (`stack.pop()`); tokens are iterated in **vocabulary insertion order**.
2. **Per-token walk**: a token advances through the DFA byte by byte (UTF-8); the moment any byte falls to `sink_state` the whole token is discarded — even if its prefix was legal. A token may stop at a non-match intermediate state (as long as that state has some non-sink DFA edge, i.e. it is "useful").
3. **Usefulness filter**: an arrived `next_state` is kept only if it is a match state, or some byte leads out of it (not to sink). If useful, **all ids** of that token (one token, many ids) get an edge `current ->(id)-> next_state`, and `next_state` is pushed if unseen.
4. **Final states**: only match states that got enumerated are final; a match state never reached by any token (and not state 0) does not count.
5. **eos self-loop**: after enumeration, each final state gains an `eos_token_id -> itself` edge. eos appears only in these self-loops, never as a regular token edge; tokens whose id list contains the eos id are skipped entirely.
6. **Incompatibility error**: a state that is not a match state and got no token edge means the vocabulary is incompatible with the regex — return the error dict immediately: the offending state plus `missing_tokens` (all non-sink bytes of that state in ascending order; `< 128` renders as the ASCII character, otherwise as `\xHH` two-digit lowercase hex). A match state with no token edges never errors.

## Input format

- `transitions`: `dict[state -> dict[byte -> next]]`, bytes 0..255; unwritten bytes go to `sink_state`; a missing state means all bytes go to sink.
- `match_states`: the DFA accepting set; `sink_state`: dead-state id (dead/quit merged abstraction); `vocab`: `dict[token_str -> list[token_id]]` (insertion order = iteration order); `eos_token_id`.

## Task

Implement `solve(transitions, match_states, sink_state, vocab, eos_token_id)` returning:

```python
{
  "initial_state": 0,
  "final_states": [...],      # ascending
  "transitions": {...},       # including eos self-loops
  "incompatible": None,       # or {"state": s, "missing_tokens": [...]}
}
```

## Example

```
Input:  transitions = {0: {ord('a'): 1}, 1: {ord('b'): 2}},
        match_states = {2}, sink_state = 3,
        vocab = {"a": [0], "b": [1]}, eos_token_id = 4
Output: final_states = [2],
        transitions = {0: {0: 1}, 1: {1: 2}, 2: {4: 2}},
        incompatible = None
```

## Constraints

- DFA state ids are non-negative and include 0; tokens are non-empty strings; `eos_token_id` may appear in some token's id list (that token is then skipped entirely).
- On error the official code raises immediately and the whole Index is unobservable — just return the error dict.
