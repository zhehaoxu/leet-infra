# WordPiece Greedy Longest Match with Word-Level UNK

## Task

Implement `solve`: given raw text and a WordPiece vocabulary, emit the subword sequence the BERT family would produce, together with token ids, code-point offsets and the word each piece came from.

WordPiece is the third mainstream subword route after BPE and Unigram, and the whole algorithm is a few dozen lines — yet interview answers rarely get all four conventions right at once: **(a)** each position takes the **longest** hit, not the shortest piece and not the fewest pieces; **(b)** a continuing piece is looked up **with the continuation prefix attached** (BERT uses `##`), so the bare `cat` sitting in the vocabulary can never be used inside a word; **(c)** the moment some position has no hit at all, the **entire word** collapses to a single `[UNK]` and every piece already cut from it is thrown away — this is not per-character `[UNK]`; **(d)** a word longer than the length limit collapses to `[UNK]` outright, without even trying to split. The judge also asks for code-point offsets and the word→piece map, which are exactly the two columns downstream word-level work (NER, span extraction, aligned annotations) consumes.

## Input

- `text`: `str`, the raw text, $0 \le |\mathit{text}| \le 4000$ code points.
- `vocab`: `list[str]`, $1 \le |\mathit{vocab}| \le 20000$, the list index is the token id; every entry is non-empty and entries are distinct.
- `unk_token`: `str`, must occur in `vocab`.
- `continuation_prefix`: `str`, may be empty and may be several characters long.
- `max_input_chars_per_word`: a Python `int` (not `bool`), $\ge 1$.

Pre-tokenization has a single rule: split `text` on Python whitespace (maximal runs of characters satisfying `str.isspace()` are separators, and leading or trailing whitespace yields no empty word); the resulting order gives words their numbers $0,1,2,\dots$. No lowercasing, no accent stripping, no punctuation splitting, no special-token handling. Inputs are never mutated; invalid input raises `ValueError`.

## Exact rules

Let $t$ be word number $w$, with $|t|$ code points, occupying the global half-open span $[a,b)$ inside `text`.

**Length short-circuit**: if $|t| > \mathit{max\_input\_chars\_per\_word}$, the word contributes exactly one piece, $\mathit{unk\_token}$, with offset $[a,b)$ and `word_id` $w$. No splitting is attempted.

**Greedy longest match**: otherwise start from $\mathit{start}=0$ and repeatedly take

$$
\mathit{piece}(\mathit{start})=\arg\max_{\mathit{end}\,>\,\mathit{start}}\
\bigl\{\,|\mathit{cand}(\mathit{start},\mathit{end})| : \mathit{cand}(\mathit{start},\mathit{end})\in \mathit{vocab}\,\bigr\},
\qquad
\mathit{cand}(\mathit{start},\mathit{end})=
\begin{cases}
t[\mathit{start}:\mathit{end}], & \mathit{start}=0,\\
\mathit{prefix}+t[\mathit{start}:\mathit{end}], & \mathit{start}>0,
\end{cases}
$$

append the hit verbatim (prefix included), set $\mathit{start}\leftarrow\mathit{end}$, and stop when $\mathit{start}=|t|$.

**Word-level UNK**: if some $\mathit{start}$ has no hit at all, **discard every piece already produced for this word** and keep a single $\mathit{unk\_token}$ with offset $[a,b)$ and `word_id` $w$. Equivalently: a word is either fully covered by vocabulary pieces or contributes one lone `[UNK]` — the output never mixes "a few pieces plus an `[UNK]`" for one word.

**Greedy is not search**: longest match neither minimises the piece count nor succeeds whenever *some* covering split exists. If the vocabulary has `cat` but not `##cat`, then even though `ha`, `##sh` and `cat` together tile the whole word, the result is still a single `[UNK]`.

## Returns

The 4-tuple `(tokens, ids, offsets, word_ids)`, each of length $P$ (possibly 0):

- `tokens`: `list[str]`, the pieces verbatim (continuations **carry** the prefix; the unknown piece is `unk_token` itself).
- `ids`: `list[int]`, the index of `tokens[i]` inside `vocab`.
- `offsets`: `list[list[int]]`, each entry `[start, end)` in **code points** of `text` — not UTF-8 bytes, not UTF-16 units.
- `word_ids`: `list[int]`, the word number ($0$-based) each piece belongs to.

Empty or whitespace-only text returns four empty lists. Everything is compared exactly.

## Example

`vocab = ["[UNK]", "the", "un", "##af", "##fable", "bow", "##els", "a", "##a", "ha", "##sh", "cat", "##t"]`, `unk_token = "[UNK]"`, `continuation_prefix = "##"`, `max_input_chars_per_word = 10`, `text = "the unaffable bowels hashcat cat"`:

```python
# words      the(0,3) w0 | unaffable(4,13) w1 | bowels(14,20) w2 | hashcat(21,28) w3 | cat(29,32) w4
# tokens   ["the", "un", "##af", "##fable", "bow", "##els", "[UNK]", "cat"]
# ids      [    1,    2,     3,        4,     5,      6,       0,     11]
# offsets  [[0,3], [4,6], [6,8],   [8,13], [14,17], [17,20], [21,28], [29,32]]
# word_ids [   0,    1,     1,       1,      2,       2,       3,       4]
```

Word by word: for `unaffable` the longest hit at `start=0` is `un` (none of `una`, `unaf`, …, `unaffable` is in the vocabulary), at `start=2` it is `##af` (`##aff` and `##affa` are absent), and `start=4` hits `##fable`. `hashcat` picks up `ha` then `##sh`, and at `start=4` only `##cat`, `##ca`, `##c` may be looked up — the bare `cat` in the vocabulary is a word-initial form and is unusable here, so the word collapses to `[UNK]` and the two earlier pieces are discarded; the standalone `cat` right after it succeeds because `start=0` adds no prefix. Change `max_input_chars_per_word` to 3 and `bowels` (6 code points) short-circuits to `[UNK]` with `offsets` `[14, 20]` and `word_ids` `2`.
