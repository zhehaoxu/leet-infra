# notes — ocr-rec-norm-edit-dis (order 687)

## Upstream

- Source: `PaddlePaddle/PaddleOCR` `ppocr/metrics/rec_metric.py`, `RecMetric.__call__` (the rapidfuzz version; older versions used python-Levenshtein, semantically equivalent). Defaults: `ignore_space=True`, `is_filter=False`, `eps=1e-5`.
- `rapidfuzz.distance.Levenshtein.normalized_distance` semantics (pinned from `Levenshtein_py.py`): `dist / maximum` where `maximum = max(len(s1), len(s2))` under weights (1,1,1); **returns 0 when `maximum == 0`** (`dist / maximum if maximum else 0`), i.e. empty-vs-empty ned=0, empty-vs-nonempty ned=1.0.
- Normalization keeps only `string.digits + string.ascii_letters` then lowercases (`_normalize_text`); CJK characters are stripped. Official order: `replace(" ", "")` first, then normalize (the filter already drops spaces; order has no behavioral difference — copied as-is).
- Classic Levenshtein is not Damerau: `"ab"→"ba"` costs 2. Transpositions live in rapidfuzz's `DamerauLevenshtein` class, not used here.
- Boundary vs `eval-*` (bleu/rouge/shingle): this problem is **character-level per-line** edit distance with a per-sample mean; HF `evaluate`'s CER is corpus-level `Σdist/Σlen` — different ledger, do not conflate.

## Judging

- solve inputs: pairs ([[pred, target], ...], may be empty), ignore_space, is_filter. Returns rows/acc/norm_edit_dis.
- Oracle: `test_cases._py` implements Levenshtein independently via recursion + `lru_cache`, with the semantics replicated (`dist/maximum if maximum else 0`); 200 random cross-checks (insert/delete/substitute perturbations + random switch combinations).
- Anchors: kitten/sitting dist=3, ned=3/7; `abcd/ab` ned=0.5 (max denominator; len(target) would wrongly give 1.0); empty-vs-empty ned=0 and exact=True; empty-vs-nonempty ned=1.0; `ab→ba` dist=2 (Damerau trap); `A-1 x!` vs `a1X` equal after filtering; empty batch acc=0.0, norm_edit_dis=1.0 (counterintuitive direct formula behavior); all-correct batch acc = n/(n+eps) strictly below 1.
- Invalid inputs: pairs not a list / elements not 2-item pairs / non-str entries, non-bool switches → ValueError.

## Mutations (13, all caught)

ned-denominator-target, ned-empty-one, ignore-space-off-pred/target, is-filter-no-lower, is-filter-no-filter, lev-no-substitute, lev-diag-free, acc-no-eps, ned-no-1minus, rows-reversed, exact-before-filter, ned-round2.

## Red lines

No corpus-level HF CER/WER (distinguished from this per-sample mean); no diff alignment output (editops excluded); no real model or eval-framework IO — strings are explicit inputs.
