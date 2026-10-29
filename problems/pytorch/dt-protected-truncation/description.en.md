# Budget Truncation with Protected Special Tokens

## Task

Special tokens have already been inserted into a sequence pair. Given token IDs, special-position flags, and sequence ownership, truncate ordinary tokens to a total length budget while preserving every special position and keeping all fields aligned. Do not tokenize, insert templates, or pad batches.

The three lists have equal length, 0..200. Sequence IDs are -1 at special positions and 0 or 1 elsewhere; the sequences may interleave. Token IDs are integers in 0..10^6. Protection depends only on `special_mask`: an ordinary token with the same numeric ID as a special token is not protected.

## Budget allocation

`max_length` must be a nonnegative Python integer, excluding bool. For original length L, delete `max(0,L-max_length)` ordinary tokens.

- `only_first`: delete only from sequence 0.
- `only_second`: delete only from sequence 1.
- `longest_first`: repeatedly decrement the side with **more remaining ordinary tokens**, choosing sequence 1 on a tie. Special positions do not count toward either side.

Raise `ValueError` if protected positions exceed the budget or the selected side cannot supply enough deletions. Unknown strategies, unknown retention modes, and invalid budgets also raise `ValueError`, even when no truncation is needed. All other inputs are valid.

## Position selection

First determine each side's retained quota q, then select positions in that side's original order:

- `head`: retain the first q.
- `tail`: retain the last q.
- `middle`: **remove the middle**, retaining the first `ceil(q/2)` and last `floor(q/2)`. Do not retain a central window.

A zero quota keeps no ordinary tokens from that side. Merge the selected positions and every special position by ascending original index. Do not move special tokens, renumber sequences, or mutate inputs.

Return a dictionary with exactly five Python-list values: `input_ids`, `special_mask`, and `sequence_ids` contain retained fields; `kept_indices` lists ascending original positions; `removed_counts` is `[removed_from_0,removed_from_1]`.

## Executed example

```text
input_ids    = [101,10,11,12,13,14,102,20,21,22,23,102]
special_mask = [true,false,false,false,false,false,true,false,false,false,false,true]
sequence_ids = [-1,0,0,0,0,0,-1,1,1,1,1,-1]
max_length = 8
strategy = "longest_first"
keep = "middle"
```

```text
input_ids      = [101,10,11,14,102,20,23,102]
special_mask   = [true,false,false,false,true,false,false,true]
sequence_ids   = [-1,0,0,0,-1,1,1,-1]
kept_indices   = [0,1,2,5,6,7,10,11]
removed_counts = [2,2]
```

Three protected positions consume budget. Ordinary quotas decrease from (5,4) to (3,2). The first side retains two leading and one trailing token; the second retains one at each end.

## Scope

This is an explicitly specified protected-position and pair-budget algorithm. `middle`, protection on an already templated sequence, and arbitrary interleaving are exercise conventions, not claims of equivalence to a tokenizer's complete truncation API. Unlike `sequence-pack-pad`, this does not pack multiple sequences; unlike few-shot budgeting, it does not select entire demonstrations.
