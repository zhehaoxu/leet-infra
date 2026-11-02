# notes (ag-self-instruct-filter)

## Source

- Primary: `yizhongw/self-instruct`, `self_instruct/bootstrap_instructions.py` (main branch, fetched through the GitHub contents API and checked line by line)
  - `post_process_gpt3_response` (41-70): `response is None or finish_reason == "length"` returns `[]` outright; line 44 `re.split(r"\n\d+\s?\. ", ...)`; lines 47-48 `\s+` collapse + `strip()` + `strip().capitalize()`; line 52 `len(inst.split()) <= 3 or > 150`; line 55 the twelve `\b(w)\b` IGNORECASE keywords; line 61 `startswith("Write a program")`; line 64 `inst[0] in string.punctuation`; line 67 `not inst[0].isascii()`.
  - Lines 195-213: `max(rouge_scores) > 0.7` drops the candidate (strictly greater), `all_instructions = seed + machine` feeds `np.argsort(rouge_scores)[-10:][::-1]` for `most_similar`, `avg = np.mean(rouge_scores)`, and acceptance does `machine_instructions.append(inst)`, growing the pool.
  - `encode_prompt` (19-29): two possible headers + `f"{idx+1}. {instruction}\n"` + a trailing `f"{len+1}."`, with each entry first getting `\s+` collapsed, `strip()`ed and `rstrip(":")`ed.
- Upstream `sample_machine_instructions` (32-34) is `random.sample` and no randomness is in scope; `make_gpt3_requests` is skipped entirely.

## Normalizations and boundary

- **Similarity as an explicit input**: upstream computes ROUGE-L inside the loop. This problem does not recompute it (`eval-rouge-score` already covers ROUGE-L scoring); the caller supplies the vector of the `i`-th candidate against the current pool, and the problem tests only the threshold decision, pool growth, top-k and the mean. A useful side effect is a hard constraint: **rejected candidates never join the pool**, so later vector lengths do not grow - a very easy thing to get wrong.
- **Ordering must be pinned here**: `np.argsort` defaults to quicksort, which is **not stable**, so on ties which entries make the top 10 is implementation-defined. This problem normalizes it to "sort by `(score, index)` ascending (equivalent to stable ascending), take the last `top_k`, reverse", which makes larger indices come first on ties and is testable.
- **Empty pool**: upstream `max([])` raises and `np.mean([])` returns `nan`. This problem normalizes it to "accept, `max_similarity=None`, `avg_similarity=0.0`" and states it in the statement.
- The `capitalize()` side effect is the core trap of this problem: it lowercases the whole string and then uppercases the first letter, so (1) `WRITE A PYTHON SCRIPT TO COUNT PRIMES` becomes `Write a python script to count primes`; (2) `Write A Program To Print Hello` is caught by `startswith("Write a program")` (likewise for `WRITE A PROGRAM ...`) - which only holds because `capitalize()` runs before the prefix test.
- `\b` interacting with CJK, measured: a `file` immediately followed by a CJK character does not hit `\bfile\b` (both sides need a boundary), `profile` does not hit `\bfile\b`, but `Go  To the market` does hit `\bgo to\b` (case-insensitive) because `\s+` collapse turns it into `Go To the market` first.
- The split regex `\s?` also allows only **one** whitespace: `\n9  . x` does not split, `\n9\n. x` does (a newline counts as that one whitespace), and `\n9.x` does not (a space must follow the dot). These three are what `m-split-star` / `m-split-no-space` hinge on.

## Mutations

All 28 are CAUGHT (`.muts-si.json`, deleted after the run):

- Splitting: `\s?` to `\s*`, dropping the space after the dot;
- Normalization: no whitespace collapse, no `capitalize()`;
- Filtering: `<=3` to `<3`, `>150` to `>151`, dropping the `go to` keyword, `\b` to no boundary, punctuation gate disabled, ASCII gate disabled, `startswith` lowercased (never matches after `capitalize()`);
- Truncation: ignoring `finish_reason`, ignoring `response is None`;
- Dedup: `>` to `>=`, no threshold check at all, accepted candidates not appended, the `sims` count check disabled;
- Top-k: no reversal (ascending), no `top_k` slice;
- Statistics: `avg` replaced by `max`, `nan` for the empty pool;
- Prompt: trailing colon not stripped, whitespace not collapsed, header always the classification one, numbering starting at 0, final line off by one.

**Known equivalent (not listed)**: changing the top-k sort key from `(vec[i], i)` to `vec[i]` - Python's `sorted` is already stable, so ties still come out in ascending index order and the semantics are identical, so it is not worth a mutant.

The only first-round SURVIVOR was `m-split-star`: the original cases only had "one space" and "one tab" splits, so `\s?` to `\s*` had nothing to expose; adding "two whitespace characters before the dot do not split" and "a newline acting as that whitespace still splits" caught it.

## Judge notes

- Tolerance: `avg_similarity` compares at 1e-12; the scores themselves stay exactly equal (they are copied straight from the input).
- `most_similar` is an **ordered dict**, so assertions must compare `list(dict)` order rather than the key set - that ordering is the only observable that separates ascending / descending / tie direction.
- The `reason` string is part of the contract (`empty` / `too_short` / `too_long` / `keyword` / `write_a_program` / `punct` / `non_ascii`); asserting only `kept` throws away which gate fired.
