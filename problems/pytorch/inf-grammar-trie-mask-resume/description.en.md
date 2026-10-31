# Constrained Decoding: Token Prefix Trie and Incremental Mask-Reuse Ledger

## Problem

xgrammar (mlc-ai/xgrammar) does two preprocessing/incremental things for grammar-constrained decoding: `TokenizerInfo` organizes the vocab into a pseudo-trie of "sorted array + subtree ranges" (`sorted_decoded_vocab` + `trie_subtree_nodes_range`), and `GrammarMatcher::FillBitmaskForStates` walks uncertain tokens in lexicographic order applying the LCP three-branch prefix-reuse logic. This problem transcribes that ledger, abstracting the Earley state machine into a deterministic "target-string prefix closure" byte state machine. Boundary with `inf-grammar-mask-dfa`: that problem judges tokens against a single-step DFA state; this one is **cross-token prefix reuse and whole-subtree skipping**.

## Solution

**Layer 1 (TokenizerInfo ledger)**:
1. Classification: tokens hit by the official detection set `DETECTION_STOP_TOKENS` (8 entries: `</s>`, `<|end_of_text|>`, `<|eot_id|>`, `<|endoftext|>`, `<eos>`, `<|eos|>`, `<end_of_turn>`, `<｜end▁of▁sentence｜>`) or in explicit `stop_token_ids` are stop (the detection set is **not** used in explicit mode); `token == ""` is special; the rest enter sorted_decoded_vocab.
2. Sorting: by token string in lexicographic order; ties pinned to ascending token id (the official `std::sort` is unstable; a stable convention is pinned in the statement).
3. `trie_subtree_nodes_range[i]` (value semantics: subtree = `[i, range[i])`): the official monotone stack transcribed line by line, with the pop condition `cur.find(top) == npos` — **a substring-containment test rather than a prefix test; the official quirk is preserved verbatim** (stack top `xab` meeting `xxab`: `find("xab")==1≠npos` so it is not popped and the range is wrongly stretched to [2,2)).

**Layer 2 (incremental-reuse ledger)**: uncertain tokens are walked in ascending sorted index (input gives token ids, converted internally):

- **subtree-skip**: `cur_idx < last_rejected_range` → the whole subtree is rejected (the official kRejected delta);
- **LCP short-circuit**: if the longest common prefix with prev_token satisfies `lcp > prev_matched` (cur still shares the prefix past prev's failure position), the state machine's determinism ⇒ it must fail at the same byte → reject and set `last_rejected_range = subtree_range[cur]`;
- **rollback-and-resume**: `lcp < prev_matched` → `prev_matched = lcp` (the official `PopLastStates`), resuming from the shared prefix; `lcp == prev_matched` resumes in place;
- the walk advances byte by byte from `prev_matched` (`advance(s)` ⇔ s is a prefix of some target); failure records walk-rejected and sets the range, while `prev_matched` stays at the last successful length (no rollback on failure, enabling the next token's short-circuit);
- prev_token is **updated every round** (on accept and reject alike; the official atomic-retry nullptr path is out of scope).

## Example

```
Input:  vocab = ["xab","xxab"], targets = ["xa","xxab"], uncertain = [0,1]
Output: trie_subtree_nodes_range = [2,2] (quirk; correct semantics would be [1,2]);
        ledger = [walk-rejected, subtree-skip], accepted = []
        (xxab is independently acceptable but over-rejected by the quirk-stretched subtree)
```

## Constraints

- encoded_vocab/targets are lists of str; uncertain_token_ids is a list of ints — out-of-range, containing stop/special, or bool → `ValueError`; stop_token_ids out-of-range/bool → `ValueError`.
- Return sorted_token_ids / sorted_tokens / trie_subtree_nodes_range / stop_token_ids / special_token_ids / accepted_token_ids / ledger (per-token branch ∈ accepted | walk-rejected | lcp-short-circuit | subtree-skip, with lcp and matched_from).
