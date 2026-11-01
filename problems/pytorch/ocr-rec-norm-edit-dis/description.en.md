# Recognition Metrics: Per-line Normalized Edit Distance Ledger

## Problem

When evaluating an OCR recognition model, each sample yields a pair `(pred, target)` of strings. PaddleOCR's `RecMetric` computes a per-line normalized edit distance (NED) and aggregates it into two metrics: `acc` and `norm_edit_dis`. This problem transcribes that ledger: given a list of line pairs and two preprocessing switches, output each line's edit distance, normalized edit distance, exact-match flag, and the aggregated metrics.

## Solution

The ledger is pinned by the spec (faithful to upstream):

1. Per-line preprocessing order: with `ignore_space=True` (default) both sides first run `replace(" ", "")`; with `is_filter=True` they are then normalized — `filter` keeps only characters in `string.digits + string.ascii_letters`, followed by `lower()`;
2. `dist` is the classic Levenshtein distance (insert/delete/substitute weights all (1,1,1), **no** Damerau transposition: `"ab"→"ba"` costs 2);
3. `ned_i = dist / max(len(pred), len(target))` — the denominator is the **longer** string's length, not `len(target)`; when both are empty the denominator is 0 and `ned_i = 0`;
4. `exact = (preprocessed pred == target)` (strict equality);
5. Aggregation: `acc = exact_count / (n + eps)`, `norm_edit_dis = 1 - Σned_i / (n + eps)`, with `eps = 1e-5`. For an empty batch (n=0) the formulas apply directly: `acc = 0.0`, `norm_edit_dis = 1.0`.

Returns `{"rows": [{"pred", "target", "dist", "ned", "exact"}, ...], "acc", "norm_edit_dis"}`, where `pred`/`target` inside `rows` are the preprocessed texts.

## Constraints

- 0 ≤ n ≤ 10^3 lines, each line length ≤ 500, arbitrary Unicode strings.
- `pairs` not a list, elements not 2-item pairs / containing non-strings, switches not bools → `ValueError`.
