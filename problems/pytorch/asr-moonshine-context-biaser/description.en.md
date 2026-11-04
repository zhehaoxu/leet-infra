# Moonshine context biaser trie walk

Moonshine's (moonshine-ai/moonshine) `ContextBiaser` (`core/context-biaser.cpp`)
pushes greedy argmax toward key terms (jargon, product names, proper nouns) at
decode time — the **zero-retraining logits-bonus ledger**: tokenized key terms
live in a prefix trie, decoding tracks which trie nodes the emitted prefix has
reached, and just before the argmax a bonus lands on the tokens that would
continue one of those active paths. The official doctest
(`context-biaser-test.cpp`) pins every behavior; this problem checks the same
numbers.

## Mechanism

### 1. Trie registration (build)

- root = node 0; a term's token sequence hangs children one by one; **shared
  prefixes reuse nodes** (`Kubernetes` = `|Ku` + `bern` + `etes`, no duplicates
  across terms sharing a prefix);
- registering the same sequence twice: `sequence_count` +1 but **no new nodes**;
- empty sequences are not registered — an empty biaser is an exact no-op.

### 2. Variant rule (variants)

The tokenizer marks word starts with U+2581 (▁), so the mid-sentence and
utterance-initial forms of a term have different first tokens and both belong
in the trie: after trimming, empty → `[]`; already anchored with ▁ → `[term]`
(the caller decides); otherwise `[term, " " + term]`.

### 3. Logarithmic depth ramp (bonus_for_depth)

`bonus = boost × (1 + ln depth)` (depth ≥ 1, default boost = 2.0). **Completing
is worth more than starting**: greedy decoding cannot take back an emitted
token, so a flat bonus would make short prefixes of key terms fire on unrelated
audio; the ramp keeps starting cheap and completing strongly rewarded. depth 0
→ 0.

### 4. apply: proposals with dedup-max

Every child of an active node is proposed with `bonus_for_depth(node.depth +
1)`; out-of-range tokens (< 0 or ≥ vocab) are skipped; **when overlapping key
terms propose the same token, the larger bonus wins, never stacked** (the
official comment: a token shared by many terms must not be boosted out of all
proportion). An empty biaser (sequence_count == 0) is a no-op. Returns
`{token: bonus}`.

### 5. advance: the walk with an always-active root

`next_active = [root] + children matched from the active nodes`:

- **the root always stays active** — a term can start part-way through another
  (while `[10, 11]` is mid-match, `[11, 20]` may still start at 11);
- a mismatched token drops the walk **back to the root** (abandoning restarts,
  no partial-match residue).

### 6. The decode loop (bias_walk)

Per step: the `apply()` bonuses land **before** the argmax → pick the token →
`advance(chosen token)`. reset returns to `[0]` (every decode pass restarts
from BOS). During speculative verification the walk follows the
**teacher-forced prefix**, not the predictions (position t is conditioned on
the prefix itself) — a pitfall the official comments call out explicitly.

## Judging domain

Single entry `solve(mode, **kw)`, seven modes: `variants` / `build` /
`bonus_for_depth` / `apply` / `advance` / `bonus_for_token` / `bias_walk`. Pure
Python + math (no torch RNG); every value is pinned by the official doctest
(boost = 2 → depth-1 = 2.0, depth-2 = 2(1+ln2), depth-3 = 2(1+ln3)).
Tokenization itself is out of scope (token sequences are explicit inputs); the
speculative-verification divergence ledger is a separate streaming problem.
