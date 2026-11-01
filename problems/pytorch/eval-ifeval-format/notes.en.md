# notes — eval-ifeval-format (order 709)

## Provenance

- Source: `google-research/google-research` subdirectory `instruction_following_eval/instructions.py` (original 2023 commit `e4c89b6404` diffed line-by-line against the 2026 master — no differences in this problem's three checkers). Registry keys: `detectable_format:json_format` / `detectable_content:postscript` / `combination:two_responses`. Paper 2311.07911.
- **capital_word_frequency and sentence counting are excluded**: the official code has used `nltk.word_tokenize` (with internal punkt sentence splitting) since the original 2023 release, and `nltk.data.load("nltk:tokenizers/punkt/english.pickle")` for sentence counts. The judge environment lacks NLTK data, so the semantics cannot be reproduced exactly. The "`\b[A-Z]+\b` regex" note in order 665 is a mis-recollection — the official implementation never used it.

## Judge surface

- solve takes instruction_ids/kwargs_list/response and returns results (per-item observed + pass) plus all_pass, matching order 665's interface.
- Oracle: `test_cases._py` is an independent rewrite — json strips ticks via `re.sub(r"^```(?:json|Json|JSON)?", ...)`, postscript uses `splitlines` + per-line `re.search` (instead of a MULTILINE findall; random cross-checking avoids `\r` to keep them equivalent), two_responses uses list comprehensions + `parts[1:-1]` empty-segment detection; 120 random cross-checks.
- **observed is pinned by this problem** (official returns bool only): json_format = bool; postscript = MULTILINE match count (= matching lines); two_responses = total non-empty segments (all non-empty segments even when a middle empty segment triggers the official early return — the collected set is equivalent).

## Anchors (hand-computed and verified against official logic case by case)

- json_format: the four prefixes ```json / ```Json / ```JSON / ``` each strip once, `removesuffix("```")` strips once; `"42"/"null"/"true"/"NaN"/"Infinity"` valid, `"True"` invalid (hand-computation corrected: Python json only extends to NaN/Infinity, not capitalized booleans); single quotes / trailing commas / empty string / bare ``` / six backticks leaving ```{}``` / `"```json{...}```json"` (suffix not a bare tick) all fail; surrounding whitespace passes; NBSP (`\xa0`) is only stripped where the outermost strip or the final strip reaches — a tick-wrapped NBSP (` ```json\xa0{}\xa0 `) relies solely on the final strip, so dropping it catches that case.
- postscript: the P.P.S branch `\s?p\.\s?p\.\s?s` allows **exactly one whitespace each** (two spaces fail); the P.S. branch `p\.\s?s\.` requires a dot right after `s`; branch comparison uses the original marker casing — lowercase `"p.s."` falls to else (`p\.s\.` has no `\s?`, so `"p. s. x"` fails while the capitalized marker passes); **the marker is not re.escape'd**: in the spliced else pattern `.` is a wildcard (marker `"P.S"` matches `pxs hello`; marker `"P.P.S."`'s `p.p.s..` has two consecutive wildcard dots that eat two spaces, so `"P.P.S  x"` passes); MULTILINE means any line can match, each matching line counted.
- two_responses: exactly two segments that differ after strip (no lowering — `"Same******same"` passes); three segments fail; leading/trailing empty segments skipped (`"******a******"` leaves one segment, fails; `"x ****** y ******"` still has two, passes); a middle empty segment fails immediately; 5 asterisks don't split, 7 leave `["", "*"]`, 12 leave two empties — all fail.
- Combined semantics: both checkers see the same response — postscript can only hide inside a JSON string value for a json+postscript double pass; `"```json\n{...}\n```\nP.S. ..."` is unparseable as a whole → json fails.

## Mutation log (17/17 caught)

json-prefix-Json / json-prefix-JSON (casing variants), json-suffix-tick, json-tick-global-strip (global replace vs single strip), json-container-only (dict/list only → scalars fail), json-catch-narrow (except TypeError misses), ps-sstar (\s?→\s*), ps-branch-swap, ps-no-multiline, ps-no-lower, ps-escape-marker (adding re.escape breaks the wildcard quirk), ps-pass-ge0, two-lower-eq, two-no-strip-eq, two-len-ge1, two-inner-empty-ok, two-star5.

One equivalent mutation recorded: swapping the removeprefix order of `"```Json"` and `"```JSON"` — the two patterns are case-exclusive, order-independent (SURVIVED confirmed). `json-final-strip-drop` was initially judged equivalent but is not (caught by the tick-wrapped NBSP case).

## Red lines

No model calls, no NLTK-dependent families (capital_word_frequency/sentences), no strict/loose markdown-stripping variants; boundary with order 665: this problem only covers structure checkers (JSON wrapping / postscript / dual responses), no keyword or counting overlap.
