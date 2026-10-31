# notes — inf-outlines-token-index (order 682)

## Source

- `dottxt-ai/outlines-core` `src/index.rs` (GitHub API contents + base64; the main repo dottxt-ai/outlines moved all FSM logic into outlines-core — its main branch only has `tests/types/test_to_regex.py` left).
- Key implementation points:
  - `DFA::new(regex)` (regex_automata) → `universal_start_state(Anchored::Yes)` is the start.
  - Main loop `while let Some(current) = next_states.pop()` — **LIFO**, not BFS; `seen` initially contains the start.
  - Token loop `for (token, ids) in vocabulary.tokens().iter()` — Rust HashMap iteration order is unspecified; **the problem pins vocabulary insertion order** as the determinized contract.
  - Byte-by-byte `dfa.next_state`; on `is_dead_state || is_quit_state` → `continue 'token_loop` (whole token discarded). dead and quit are handled identically → merging them into one sink state is a lossless abstraction (stated in the description).
  - **Usefulness check** (with a global cache `is_useful_state_cache`): `is_full_match_state (next_eoi is match) || check_is_intermediate_state (some byte-class representative leads out, non-dead)`. Byte-class representatives are equivalent to all 256 bytes (same-class bytes share transitions). Short-circuit order: match first, intermediate second.
  - **Final test**: `dfa.is_match_state(dfa.next_eoi_state(current))` — reaching match on eoi ⇔ the state itself is a match (equivalent for anchored semantics), abstracted to `current in match_states`.
  - **IncompatibleVocabulary**: `!has_valid_transitions && !dfa.is_match_state(current)`; missing_tokens branches on `byte.is_ascii()` (ascii → char, else `\x{:02x}`). Note the criterion is **`byte.is_ascii()` (< 128)**, not "printable" — control chars like \x00 also render as raw characters.
  - **eos self-loop**: after the loop, every final state gains `eos -> final`; the `next_state(eos)` API always returns None (even though the edge exists in transitions) — the data-layer/API-layer "contradiction" is official design; this problem tests only the data layer.

## Abstractions vs. the official code (all pinned in description/notes)

- The regex→DFA step is not tested (no Python equivalent of regex_automata); the DFA is an explicit transition table: `dict[state -> dict[byte -> next]]` + unwritten bytes go to sink + a `sink_state` parameter. State 0 = anchored start.
- eoi semantics abstracted into a given match set.
- vocab is `dict[str, list[int]]`: insertion order = the determinized stand-in for the official token iteration order; one token with many ids shares one transition (official `for token_id in ids` inserts all).
- Errors return a `{"incompatible": {...}}` dict instead of raising, keeping the output deterministic.

## Core ledger points (each nailed by a test)

1. **LIFO order decides which state errors first**: two incompatible states — the one popped first reports (check_incompatible_dfs_order; caught by mutation pop-fifo).
2. **Dead ends never enqueue**: a non-match state with no DFA out-edge is not useful → never enqueued → never errors, never indexed; tokens reaching it are discarded wholesale (check_dead_end_not_error; caught by push-not-useful: enqueuing it makes it pop and falsely report incompatible).
3. **Unreached match states are not final** (check_unreached_match_not_final; caught by final-all-match).
4. **Match states are not absorbing**: the walk continues past a match; regular token edges from a match state remain and coexist with the eos self-loop (check_over_match_continue / check_initial_match).
5. **A token may stop at a useful non-match intermediate state and still get its edge** (check_stop_at_intermediate).
6. **Tokens containing the eos id are skipped entirely**; eos appears only as self-loops (check_eos_token_skipped; caught by no-eos-skip).
7. **One token, many ids → all ids get edges** (check_multi_id_edge).
8. **UTF-8 byte-level walking**: any byte hitting sink discards the whole token; intermediate states never reached by a complete token are not indexed (check_multibyte_token_walk). **Python trap**: `"\xf0"` is the character ð (encodes to c3 b0), not the bare byte 0xF0 — hit while building the test.
9. **missing_tokens format ledger**: bytes ascending, ascii/<128 via chr, else `\xHH` two-digit lowercase hex (check_incompatible_hex_and_order; caught by miss-no-hex / miss-reversed).
10. **First incompatibility short-circuits**: after the error the whole Index is unobservable (official Err path), so error tests can only assert the incompatible dict itself.

## Mutation record

10 observable mutations all CAUGHT (`.mutate.py`):
`useful-no-match` (useful drops the match branch), `final-all-match` (finals = all match states), `no-eos-self-loop`, `eos-loop-to-init` (self-loop targets 0), `no-eos-skip`, `dead-reset-restart` (reset to 0 and continue on sink), `push-not-useful`, `drop-match-valid`, `miss-reversed` (missing descending), `miss-no-hex`.

**Equivalent-mutation analysis (first version of drop-match-valid)**: in the official code the match branch's `has_valid = True` and the error condition `... && !dfa.is_match_state(current)` are mutually redundant (belt and suspenders) — removing either alone is unobservable; both must change together to build a "match state with no token edges errors" semantic variant. The observable variant confirmed the equivalence; miss-reversed was added to round out 10 observable mutations.

## Hand-computed anchor pitfalls

- In `transitions[2]`, "0" departing from state 2 self-loops ('0'→2); it is not the state-0→1 edge — the first assertion wrongly said {3:1}.
- The str-token UTF-8 trap from point 8: a "bare 0xF0 single-byte token" is inexpressible as a str, so the test was redesigned around the full four-byte 😀 DFA.
- The first draft of check_multibyte_token_walk also tried to assert pre-error edges — on the error path the whole Index is unobservable, so assertions must land on the incompatible dict only.

## Boundaries with adjacent problems

- `inf-grammar-mask-dfa` (order 458): after regex→DFA, **runtime single-step** masking (prefix advances step by step).
- `inf-grammar-trie-mask-resume` (order 681): xgrammar's **pseudo-trie + LCP three-branch** cross-token prefix reuse.
- This problem: the **compile-time** ledger of compressing DFA×vocabulary into an index (usefulness filter, eos self-loops, incompatibility detection) — no runtime querying appears.
