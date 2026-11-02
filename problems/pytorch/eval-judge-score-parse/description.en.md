# LLM-as-a-Judge: Score-String Parsing and Single-Score Table

## Problem

A judge model emits free-form text whose last line reads `[[8, 7]]` (pairwise grading) or `[[8]]` (single-answer grading). The server does not understand the prose at all: it pulls the numbers out with two regexes, marks the row `error` on a miss, and in single mode records `-1` and drops it during aggregation.

This problem covers exactly that slice: **the four-regex score parsing** (primary / backup tiers, `\s?` meaning exactly one optional whitespace, the unsigned `\d+\.?\d*` number), the `int` / `float` distinction produced by `ast.literal_eval`, the `tie_delta` verdict in pairwise mode, and the single-mode `(model, turn)` mean table. The position-bias swap games, disagreement-as-tie and win-rate aggregation live in a separate problem (`eval-judge-swap-consistency`) and are out of scope here.

## Rules

Signature: `solve(entries, mode="pair", tie_delta=0.1)`.

**1. The four regexes** (same origin as FastChat `llm_judge/common.py`):

- Pairwise primary `\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]`, pairwise backup `\[(\d+\.?\d*),\s?(\d+\.?\d*)\]`;
- Single primary `\[\[(\d+\.?\d*)\]\]`, single backup `\[(\d+\.?\d*)\]`.

All of them search the whole string and take the leftmost hit, so the numbers may sit anywhere and trailing text is irrelevant; the backup pattern is tried **only after** the primary one fails. Three details must be reproduced exactly:

- `\s?` allows **at most one** whitespace character and `\s` follows Unicode: space, tab, newline, `\xa0` (NBSP) and `\u3000` (ideographic space) each count as one; two spaces count as zero, so primary and backup both miss.
- The number is `\d+\.?\d*`: no sign (`[[-1, 7]]` yields no score at all), no bare fraction (`.5` fails), but a trailing dot (`5.`) is accepted.
- The primary pattern requires a digit **right after** `[[`; `[[ 8, 7]]` misses both tiers, while `[[8, 7]]x` still hits (no end anchor).

Each captured substring is parsed as a Python literal: `int` without a dot, `float` with one (`5.` becomes `5.0`); a leading-zero token such as `007` is not a legal literal and raises `ValueError`.

**2. Pairwise mode** (`mode="pair"`, `entries` is a list of strings): every row is `{"pattern", "scores", "verdict"}`. `pattern` is in `{"two", "two_backup", "none"}`; on a miss `scores = None` and `verdict = "error"`; otherwise `scores` holds the two numbers, `|a - b| <= tie_delta` yields `"tie"`, and otherwise the higher A slot yields `"A"` while a higher B slot yields `"B"`. In this mode `table == {}` and `rank == []`.

**3. Single mode** (`mode="single"`, `entries` is a list of `{"model", "turn", "text"}`): every row is `{"model", "turn", "pattern", "score"}`, with `pattern` in `{"one", "one_backup", "none"}` and `score = -1` on a miss.

Aggregation first drops rows with `score == -1`, then groups by model: `turn_1` / `turn_2` are the arithmetic means of that turn's scores (`None` when the turn has no valid row) and `avg` is the mean over all valid rows. `rank` is `avg` descending with ties broken by model name ascending. A model whose rows are all `-1` does not appear in `table` at all.

## Example

```text
input: entries = ["[[8, 7]]", "[[7, 8]]", "[[9, 9]]"], mode = "pair"
output: rows = [{"pattern": "two", "scores": [8, 7], "verdict": "A"},
                {"pattern": "two", "scores": [7, 8], "verdict": "B"},
                {"pattern": "two", "scores": [9, 9], "verdict": "tie"}]
        table = {}, rank = []
```

```text
input: entries = [{"model": "a", "turn": 1, "text": "[[10]]"},
                  {"model": "a", "turn": 1, "text": "bad"},
                  {"model": "a", "turn": 2, "text": "[[4]]"}],
       mode = "single"
output: rows = [{"pattern": "one", "score": 10}, {"pattern": "none", "score": -1},
                {"pattern": "one", "score": 4}]
        table = {"a": {"turn_1": 10.0, "turn_2": 4.0, "avg": 7.0}}
        rank  = ["a"]
```

## Constraints

- `mode` must be `"pair"` or `"single"`; `tie_delta` must be a finite non-negative real and `bool` is rejected; `entries` must be a list or tuple.
- Every pairwise entry must be a `str`; every single entry must be a `dict` carrying `model` / `turn` / `text`, with a non-empty string model, `turn` an `int` of `1` or `2` (`bool` and `1.0` are rejected), and a string text.
- An empty `entries` is legal and yields empty tables; the returned keys are always `mode` / `rows` / `table` / `rank`, and the fields of the inactive mode stay empty.
- `0` is a legal score (it is not the `-1` sentinel); inputs must not be mutated; `scores` keeps the original `int` / `float` distinction.
