# notes — inf-grammar-trie-mask-resume (order 681)

## Source

- Authoritative source: mlc-ai/xgrammar main branch (fetched 2026-11). `cpp/tokenizer_info_impl.h` (fields sorted_decoded_vocab / trie_subtree_nodes_range / TokenCharCounts; the comment "A pesudo-trie. trie_subtree_nodes_range[i] stores how many nodes there are in the subtree." — **the comment says node counts but the implementation stores index bounds** (subtree = [i, range[i])); the problem follows the implementation), `cpp/tokenizer_info.cc` Impl constructor (classification + std::sort + monotone stack), `cpp/compiled_grammar_impl.h` (AdaptiveTokenMask::StoreType is now kAccepted/kRejected/kAcceptedBitset with USE_BITSET_THRESHOLD=1000 — the old kTree/kPrefix stores were refactored away; only the matcher-consumer side is taken), `cpp/grammar_matcher.cc` FillBitmaskForStates (Step 2.1 LCP three branches / Step 2.2 walk / the lower_bound subtree skip / per-round prev_token update / the atomic-retry nullptr path is out of scope).
- The DETECTION_STOP_TOKENS set (8 entries) transcribed verbatim (LLaMA2 `</s>`, LLaMA3 `<|end_of_text|>`/`<|eot_id|>`, Phi-2 `<|endoftext|>`, Gemma `<eos>`/`<end_of_turn>`, DeepSeek-V2 `<｜end▁of▁sentence｜>`, `<|eos|>`).
- **Official quirk (pinned into the problem)**: the trie pop condition uses `token.find(top) == npos` (substring containment) instead of a prefix test — stack top `xab` meeting `xxab` is misjudged as a prefix and not popped, stretching the range. Downstream, the stretched pseudo-subtree whole-subtree-skips tokens an independent walk would accept (over-rejection). Preserved verbatim with a dedicated anchor test.
- **Comment/implementation mismatch**: the header comment describes the range value as the subtree node count while the implementation stores index bounds; the problem follows the implementation.
- Ties on identical token strings are unspecified under the official std::sort — the statement pins ascending token id (stable), recorded here as a deviation.
- The Earley state machine is abstracted into a targets-prefix-closure byte machine: advance(s) ⇔ s is a prefix of some target (full match included; advancing further fails). Since the predicate depends only on the current string prefix, the LCP short-circuit's correctness (same prefix ⇒ same failure) holds under this convention.
- Out of scope (recorded here): char budget / token_char_counts (the bounded-string path), FindMidTokenExcludeEdgeRow mid-token divergence, atomic retry (PopLastStates(prev_matched_size+1)), kAcceptedBitset storage and USE_BITSET_THRESHOLD, vocab_type decoding (BYTE_LEVEL/BYTE_FALLBACK DecodeToken; the statement pins RAW semantics).

## Judging surface

- solve(encoded_vocab, targets, uncertain_token_ids, stop_token_ids=None) → 7 fields + ledger (token_id/sorted_index/branch/lcp/matched_from).
- Oracle: independently rewritten — sorting via "stable sort by token" (ties automatically keep id order), the trie quirk rewritten as `top not in tok`, lcp via os.path.commonprefix, the main loop in a while structure; full-field comparison. A second independent-walk oracle (from-zero walk per token, equivalent to the incremental result on quirk-free vocabs) is also asserted.
- Anchors: full hand-computed trie build (a/get/getter/go/gone/set/setter); the quirk range [2,2] and its over-rejection downstream; get→go rollback-and-resume (lcp=1); gone reusing goa's matched prefix (lcp==matched, no rollback); the LCP short-circuit (lcp=3>2, outside the subtree) that also sets the range so an in-subtree token becomes subtree-skip; the whole subtree skipped after "abc" fails; a range boundary (range[ab]=1 excluding ac); detection-set vs explicit stop modes (in explicit mode `</s>` etc. enter the sorted vocab); "" special vs " " not special; tie broken by id (including two identical tokens where lcp==matched means immediate acceptance); empty targets (all rejected, first failure at m=0); the ValueError set.
- Lazy-pop semantics: range[j] is assigned "the i at pop time"; one i can pop several stack elements (go pops get and getter, both = 3).

## Pitfalls (hit while authoring)

- Hand-computed lcp("get","go")=1 (g shared, e≠o), mis-computed as 2 (confused with the matched length).
- In the sorted order go comes after getter (getter is a prefix extension of get and greater), and mis-sorting makes the hand-computed range[get] 2 instead of the correct 3.
- range[abxyc]=4: the pop assigns "the i at pop time" (while processing aby, i=4), not 5 — usually "first position after the subtree" coincides, but when one i pops several stack elements they all take the same i.
- In the quirk fixture, xxab's independent-walk verdict equals the incremental one (both reject) — the ledger branch assertion is what pins it; a second fixture with xxab ∈ targets makes the accepted set itself diverge under the quirk.

## Mutation record (10/10 caught)

sort-tie-desc (ties by descending id), stop-detect-miss (a detection entry retyped), special-strip (special = strip-empty), quirk-fix (find→startswith, i.e. "fixing" the official quirk), range-plus-one (pop assignment +1), lcp-gt-no-range (short-circuit without setting the range), lcp-lt-nopop (lost rollback), prev-not-updated (prev_token kept on accept), advance-off-by-one (walk predicate shifted one byte), uncertain-no-sort. No equivalent mutations.

## Red lines

No real tokenizer/HF model, no real Earley parser, no CUDA kernels; the byte state machine is an explicit targets argument (same pattern as ag-tool-loop-breaker's sha256 fingerprint and inf-grammar-mask-dfa's DFA table). Boundary with `inf-grammar-mask-dfa` (order 484): single-step DFA mask vs cross-token prefix reuse/subtree skipping; with `inf-radix-prefix-cache`: KV-cache reuse vs token-mask reuse. Pure Python, no torch dependency.
