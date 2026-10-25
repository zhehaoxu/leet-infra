# Tokenization: Training BPE Merge Rules from Scratch

## Problem

Byte-Pair Encoding (BPE) is the tokenization foundation of mainstream LLMs such as GPT-2/LLaMA/Qwen. The core idea: start from characters and **repeatedly merge the most frequent adjacent symbol pair in the corpus** until enough subwords have been learned. This problem implements its **training** stage — greedily learning the first $K$ merge rules from a corpus with word frequencies. This is upstream of "encoding" (`bpe-encoding`): encoding simply applies the rules in this order.

## Algorithm (be sure to implement each step exactly as given, to guarantee a unique answer)

Given a "word → frequency" corpus and the number of merges $K$:

1. **Preprocessing**: split each word into characters and append the boundary symbol `</w>` at the end. Example: `"low"` → the symbol sequence `["l","o","w","</w>"]`. If two words have identical symbol sequences after preprocessing, add their frequencies.
2. **Count adjacent pairs**: for each word's **current** symbol sequence, count all adjacent bigrams `(s_i, s_{i+1})`, weighting each word's contribution by its **frequency** (a word occurring `f` times contributes `+f` to each of its adjacent symbol pairs).
3. **Choose the best pair**: take the pair with the largest count; **break ties by lexicographic order** (compare the first symbol first, then the second). Record it as a merge rule.
4. **Apply the merge**: in every word, replace all **left-to-right, non-overlapping** occurrences of that pair `a b` with the concatenated symbol `a+b` (`a+b` is direct string concatenation, so it may produce multi-character symbols such as `"er"`, `"er</w>"`). Multiple occurrences within the same word must all be merged (greedy scan: merge on a hit and skip two slots).
5. Repeat steps 2–4 for $K$ rounds in total (stop early if no adjacent pair can be counted in some round). Output these $K$ (or fewer) rules **in learning order**.

> The boundary symbol `</w>` participates in pairing as an ordinary symbol (so `"w"` and `"</w>"` can be merged into `"w</w>"`).

## Rules

Implement `solve(word_freqs, num_merges)`:

- `word_freqs`: a list of the form `[("low", 5), ("lowest", 2), ...]` (or an equivalent sequence of pairs).
- `num_merges`: the number of rules $K$ to learn.
- Return a **list of pairs** `[(first, second), ...]` (each item is the selected adjacent pair, in learning order; a tuple or a 2-element list is fine, and the judge normalizes to tuples for comparison).

## Examples

Corpus `[("low",5), ("lowest",2), ("newer",6), ("wider",3)]`:

- `num_merges=4` → `[('e','r'), ('er','</w>'), ('l','o'), ('lo','w')]`
- `num_merges=8` → the 4 above followed by `[('e','w'), ('ew','er</w>'), ('n','ewer</w>'), ('low','</w>')]`

Explanation: `('e','r')` has the largest count (`e r` in newer occurs 6 times), hence rule 0; after merging, newer contains `er`, and its combination with the trailing `</w>` gives `('er','</w>')` with count 9 (newer 6 + wider 3), which becomes rule 1, and so on.

## Constraints

- Words contain only lowercase letters; `num_merges ≥ 0`; frequencies are positive integers
- Ties **must** be broken by lexicographic order (otherwise the answer is not unique) — this is the biggest trap in this problem
- Pure Python string/dict operations, no randomness, no torch; the return order is the learning order
