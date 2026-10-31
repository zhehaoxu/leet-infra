# Self-Consistency Majority-Vote Ledger: Answer Normalization and Vote Aggregation

## Problem

Self-consistency (Wang et al. 2022, arXiv:2203.11171) is the cheapest free lunch in inference:
sample the same question N times, extract the **final answer** from each output, and take a majority
vote. The engineering pain is not the voting but **answer normalization** — "42", " 42 ", "42.",
"$42", "42%" are the same answer to a human but five different strings to a machine. Normalize too
loosely and the vote splits; too tightly and distinct answers get merged.

> To keep judging **deterministic**, `samples` has already been truncated by the caller to the answer
> segment (not the full CoT); the normalization rules are **pinned down item by item** in the problem
> statement — implementations may not improvise (e.g. mathematically merging "1/2" with "0.5").

## Rules

- Implement `solve(samples, weights=None)`:
  - `samples: list[str]`, the answer segment of each candidate;
  - `weights: None or list[float]`, per-sample vote weights of the same length, default all 1.
- **Normalization rules (applied in order)**:
  1. Strip leading/trailing whitespace;
  2. Lowercase everything;
  3. Remove thousands separators: only when the comma has **digits on both sides** ("12,345" → "12345");
  4. Strip one trailing period ("42." → "42");
  5. Normalize fractions to `"a/b"`: both `"a ÷ b"` and `"a / b"` (any spacing around the slash) become
     `"a/b"`, where a and b are positive integers with no adjacent digits (letter operands like "a/2"
     are exempt);
  6. Strip a percent sign: "42%" → "42";
  7. Strip a currency symbol: remove a leading "$";
  8. Keep all other characters as-is — **no mathematical merging**: "1/2" and "0.5" are different answers.
- **Tallying**: `tally` maps normalized answer → accumulated vote weight (dict keyed in **first-occurrence
  order**); `winner` is the answer with the largest weight (**ties go to the first occurrence**, i.e.
  strictly greater weight overwrites); `vote_share = winner weight / total weight` (0.0 when the total
  weight is 0).
- Return a dict: `normalized` (per-sample results), `tally`, `winner`, `vote_share`, `n_candidates`.
- Validation: mismatched `weights` length or a negative weight → `ValueError`.

## Example

```
Input: samples = ["1,234.", "1234", "1/2", "1 ÷ 2", "$1,234"]
  normalized: ["1234", "1234", "1/2", "1/2", "1234"]
  tally = {"1234": 3.0, "1/2": 2.0}
Output: winner = "1234", vote_share = 0.6, n_candidates = 2
```

## Constraints

- Pure Python string processing suffices (torch not needed); `samples` may be empty (return an empty
  ledger with `winner=None`)
- Weights may be 0 (all-zero gives `vote_share = 0.0` while `winner` still follows the
  first-occurrence rule) but never negative
- Rules 1→7 must be applied **in the stated order** (order matters: "$1,234" needs comma removal
  before "$" stripping)
