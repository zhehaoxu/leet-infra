# Deep Dive: Parsing and Validating Aspect-Ratio Bucket Strings

## Where it comes from

`parse_buckets_string` in HuggingFace diffusers' `src/diffusers/training_utils.py`. It is the entry point through which multi-resolution training scripts for SD3 / Flux read their `--bucket` config, and the resulting `[(h, w), ...]` is then consumed by `generate_aspect_ratio_buckets` / `find_nearest_bucket` (already covered by `diff-aspect-ratio-buckets`).

## Key ideas and pitfalls

1. **The regex only accepts a segment that is exactly two integers**: `^\s*(\d+)\s*,\s*(\d+)\s*$` demands a full-segment match, so `"512,768,1024"`, `"512,"`, `",768"`, `"512x768"`, `"-512,768"`, `"+512,768"`, `"5e2,768"` and `"512.0,768"` are all rejected. The `split(",")`-then-`int()` approach lets `"+512"` and `"-512"` through — the most common deviation.
2. **Whitelist, not blacklist**: the only allowed characters are digits, whitespace around the comma, and whitespace at both ends of a segment. Enumerating forbidden characters always misses a class.
3. **`\d` matches Unicode decimal digits**: `"٥١٢,٧٦٨"` (Arabic-Indic digits) passes and yields `(512, 768)`, since `int()` accepts them too. This is genuine upstream behaviour; the problem adds no assertion for it, but the implementation should stick to the `\d` + `int()` pair.
4. **The zero-size error message is wrapped twice**: upstream first `raise ValueError("Bucket dimensions must be positive integers.")`, and because that statement sits inside the `try`, the outer `except ValueError` re-wraps it as `Invalid integer in bucket pair '0,0': Bucket dimensions must be positive integers.`. Only reproducing that nested `try` produces the same message.
5. **Warnings are per bucket, not per edge**: the condition is `if height % 8 != 0 or width % 8 != 0:` with one `warnings.warn` and one message, so `"3,5"` emits a single warning rather than two. Splitting it into two independent `if`s is the easiest mistake here.
6. **A warning is not an error**: a bucket that is not a multiple of 8 is still returned. The rationale upstream is that some VAEs use a patch size other than 8, so it is only "might cause issues".
7. **Order and duplicates**: no sorting, no dedupe. The order of the bucket list affects `find_nearest_bucket`'s tie-break (ties go to the later bucket), so ordering is part of the semantics.
8. **Only `;` separates**: `|`, newlines and spaces do not split; `"512,768|256,256"` fails at the regex. A newline at either end of the string disappears because of the initial `strip()`.
9. **Where a whitespace-only string fails**: `"   "` is not "empty", so it does not hit `if not buckets_str`; instead `strip()` yields `""`, `split(";")` yields `[""]`, and the regex rejects it. Both paths raise `ValueError`, but through different branches.

## How the judge cross-checks

There are no tensors here, so the judge relies on two independent groups of assertions:

- **Cross-implementation comparison**: `_alt_parse` avoids regex entirely and uses `split(",")` + `str.isdigit()` + `int()`, requiring exactly the same return values on 7 valid samples as the reference and rejection on all 20 invalid ones;
- **Behavioural contract**: return types (list / tuple / int, and not bool), the message keywords for zero sizes and bad separators, the warning count and message keyword (covering "both edges misaligned still yields one", "two bad buckets yield two", and "no warning when both edges align"), `|` not being a separator, the trailing newline being stripped, and duplicates not being merged.

Seven assertion groups in total: example, return types, warnings, invalid inputs, error messages, cross-implementation comparison, and separator semantics.

## Complexity

One `strip`, one `split` and one regex match per segment over an input of length \(n\): \(O(n)\) time and \(O(k)\) space for \(k\) buckets.
