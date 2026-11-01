# IF-Eval Verifiable Instructions: Structure-Judge Ledger

## Problem

IFEval (google-research/instruction-following-eval, paper 2311.07911) evaluates instruction following via **programmatically verifiable instructions**. The earlier problem (order 665) covered six keyword/length/end-phrase/bullet checkers; this one transcribes three remaining **structure checkers** with their official `check_following` semantics: given `instruction_ids + kwargs_list + response`, emit per-instruction `observed` and `pass`, plus an aggregate `all_pass`.

## Solution

Three instruction types (verbatim official semantics, reproducible with the stdlib only):

- `detectable_format:json_format`: after `strip()`, `removeprefix("```json")`, `removeprefix("```Json")`, `removeprefix("```JSON")`, `removeprefix("```")` (each strips once), then `removesuffix("```")` (once) then `strip()`; pass iff `json.loads` succeeds (catching `ValueError`). Note JSON scalars `"42"/"null"/"true"` are valid, Python's `"True"` is not, while `"NaN"/"Infinity"` are Python json extensions and valid;
- `detectable_content:postscript`: strip the marker; lower the value, then branch on the **exact** marker string — `"P.P.S"` → `\s*p\.\s?p\.\s?s.*$`, `"P.S."` → `\s*p\.\s?s\..*$`, else → `\s*<marker.lower()>.*$`; `re.findall(..., flags=re.MULTILINE)`, observed = number of matches (any line suffices). Two official quirks must be kept: branch comparison uses the original marker casing (lowercase `"p.s."` falls to else, whose `p\.s\.` has no `\s?`); the else branch splices the marker into the regex **without `re.escape`**, so a `.` inside the marker is a wildcard (marker `"P.S"` matches `pxs hello`);
- `combination:two_responses`: `split("******")`; empty segments at index 0 or the last index are skipped, an empty segment in the middle fails immediately; pass iff exactly 2 non-empty segments whose stripped texts differ (comparison does **not** lower-case). observed = total non-empty segments.

## Constraints

- 1 ≤ len(instruction_ids) ≤ 8; ids must be in the 3-type whitelist.
- Length mismatch, unknown id, non-dict kwargs entry, missing/non-str/blank-after-strip postscript marker, non-str response → `ValueError`.
