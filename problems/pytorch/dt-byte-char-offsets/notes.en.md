# Notes: UTF-8 Boundary Search and Positional Ownership

## Reference algorithm

Let $L$ be the text's code-point length. Build cumulative UTF-8 boundaries $P$:

$$
P_0=0,\qquad P_{i+1}=P_i+|\operatorname{UTF8}(text_i)|.
$$

Each Unicode scalar occupies 1..4 bytes, so $P$ is strictly increasing. For an ordinary byte interval $[a,b)$, first validate $0\le a<b\le P_L$, then compute:

$$
s=\operatorname{bisect\_right}(P,a)-1,\qquad
 e=\operatorname{bisect\_left}(P,b).
$$

The result is the code-point interval $[s,e)$. No partial-byte decoding is needed: an interior start rounds down to its owning character, an interior end rounds up, and an end already on a boundary stays there.

Handle `(-1,-1)` separately before ordinary validation, producing `[0,0]` and `None`. Other negative, empty or out-of-bounds ranges cannot stand in for specials.

Each ordinary token is guaranteed to belong to one explicit word. Binary-search word starts for the last start not exceeding $s$ to obtain that word's index. Do not resplit on whitespace. In token order, record each word's first token index and update its endpoint to the current index plus one. An intervening special does not terminate the range.

For each character query, scan tokens in input order and return the first nonspecial offset covering it. Negative/out-of-bounds queries, uncovered characters and gaps naturally yield `None`.

## Correctness

The searches establish $P_s\le a<P_{s+1}$ and $P_{e-1}<b\le P_e$. Thus $[P_s,P_e)$ contains every byte of $[a,b)$. Removing either endpoint character would lose at least one token byte, proving that $[s,e)$ is the minimal cover.

Guaranteed containment and nonoverlapping word spans make ownership unique. For each word, its first matching token has the minimum index and its last matching token has the maximum index, yielding exactly the smallest enclosing half-open range. Character queries search in ascending token-index order, so duplicates, overlaps and nonmonotone ends cannot violate the first-hit rule.

## Complexity

Let $T,W,Q$ count tokens, words and queries. Time is

$$
O\bigl(L+W+T(1+\log(L+1)+\log(W+1))+Q(T+1)\bigr).
$$

Space including results is $O(L+T+W+Q)$. The implementation uses only standard-library `bisect`, with no tokenizer or tensor-library dependency.

## Independent oracle and tests

The oracle uses neither binary search nor cumulative-boundary conversion. It expands `byte_owner` and `byte_word` arrays character by character, enumerates every byte in each token, and derives the cover from the minimum and maximum character owners. Queries search owner sets directly; word ranges separately enumerate all matching token indices. It does not call the reference implementation.

`--selftest dt-byte-char-offsets --mode submit` ran first and passed **15/15**. Coverage includes Chinese/non-BMP characters, combining/precomposed accents, ZWJ/variation selectors, all nonempty subintervals within 1..4-byte characters, cross-character covers, exact boundaries, duplicates/overlaps, first hit, specials versus character 0, gaps/empty text, words without tokens, word ranges containing specials, explicit units containing whitespace, maxima of 100 code points/200 tokens/30 words/100 queries, 120 random inputs with seed 460, immutability and invalid byte ranges.

## Native frozen fixture

After independent tests passed, the complete native pipeline was executed using tokenizers **0.23.3** in `.venv-eval/bin/python`, rather than generating expectations from the reference solution. Minimal reproduction configuration follows; it is for local verification only, not a dependency of this exercise:

```python
from tokenizers import Tokenizer, models, pre_tokenizers
alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
backend = Tokenizer(models.BPE(
    vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
backend.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=True)
encoding = backend.encode('é😊X')
```

The vocabulary contains exactly 256 byte-level symbols, with no merges, normalizer or offset-trimming postprocessor. Pre-tokenization units have original code-point spans `[(0,1),(1,2),(2,3)]`. Frozen native results:

```text
offsets        = [(0,1),(0,1),(1,2),(1,2),(1,2),(1,2),(2,3)]
word_ids       = [0,0,1,1,1,1,2]
char_to_token  = [0,2,6]             # queries 0,1,2
word_to_tokens = [(0,2),(2,6),(6,7)]
```

Converting native tuple ranges to lists gives an exact match for all four exercise fields with one interval per original byte. The result is frozen as literals in `check_native_bytelevel_fixture`; tests import no third-party library. Both statement examples were then executed and recorded. The second example's arbitrary special insertions are an independent example of the exercise model, not claimed to come from this native pipeline without a postprocessor.

## Source verification and scope: pipeline versus converter

Local `tokenizers/__init__.pyi:279–402,931–967` was inspected, together with the matching [tokenizers 0.23.3 source archive](https://files.pythonhosted.org/packages/e0/7c/2cabb2174e772636683008f2c5621949b645da7d303c596589e84516a184/tokenizers-0.23.3.tar.gz), read in memory without installation or extraction into the project:

- `tokenizers/src/tokenizer/pre_tokenizer.rs:165–194`: normalized alignment's `convert_offsets` first restores original ranges, before byte-to-character conversion; words come from pre-tokenization splits or explicit word indices.
- The same file, `267–300`: the internal `BytesToCharOffsetConverter` looks up both endpoints in a map. Directly mapping `(0,1)` inside the bytes of `é` can yield `(0,0)`, **not this exercise's upward-rounded minimal cover**.
- `tokenizers/src/tokenizer/encoding.rs:228–292`: enclosing word-token ranges and the first half-open offset hit for character queries.

This is therefore a **restricted model of complete ByteLevel alignment without normalization**, not a literal reproduction of arbitrary spans passed to the standalone Rust `BytesToCharOffsetConverter`. Even without a normalizer, ByteLevel's byte-symbol representation carries alignment back to the original text. Skipping that stage does not make the internal converter expand every partial-byte span on its own.

The exercise explicitly supplies original byte spans, word spans and special sentinels, and permits duplicates/overlaps satisfying its contract. The native fixture establishes correspondence for its specified configuration, not equivalence for arbitrary tokenizers, normalization or wrappers. There is no UTF-16/grapheme mapping, NFC/NFD conversion, prefix-space insertion or offset trimming.
