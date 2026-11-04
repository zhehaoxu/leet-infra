# Solution Notes

## Approach

The deterministic core of the official WENGSYX/Self-Verification:

1. **main_verifier.py** (L140-259): candidate collection (float normalization + first-seen dedup), the single/multi-candidate branch, verifier score accumulation (L190-208), argmax selection and judgement;
2. **utils/question_turn_decalrative** (L811-896): number-blanked subquestion construction (numeric path L835-892) + the declarative rewrite cache (L854-864).

Out of scope: GPT sampling and regeneration, answer_cleansing / answer_cleansing_verifier (post-cleansing strings arrive as inputs), the demo texts, and the logging/statistics ledgers (accs/accs_verifier averages). The commonsensqa question-tail special case (L816-833) and the 'Answer Choices' prefix branch are also excluded (the numeric path consumes the full text).

Boundary with existing problems:
- **rs-self-consistency / rs-ttrl**: majority voting folds answers; this one scores **content self-consistency**;
- **775/776**: assembling demos and selecting questions; this one is the within-question candidate competition;
- **rs-prm-best-of-n**: step-level process rewards; this one is answer-level fill-back verification.

## Key implementation points (official conventions pinned)

- **The official regex `\d+\.?\d*` trailing dot**: `"Total is 6."` extracts `"6."` (`\.` optional, `\d*` may be empty) — blank matching is **intra-segment substring** based, and no dot-less segment contains `"6."` → no replacement. Numbers spanning periods (`-2.5`) likewise. These two "no blanking happens" behaviors are official reality, not bugs.
- **The missing number_find flag**: the official code tests `number_find == False` but never sets it True → a number appearing in multiple segments gets **replaced in all of them** (multi_segment asserts count('X')==2).
- **Where the `..` → `.` fix comes from**: ts ends with a period, and the empty trailing segment after split regenerates one extra dot on reassembly (`"6." + "." + "."`) → the official replace fixes it. Slicing ts one character short defeats this fix (a mutation point).
- **`new_question.replace(' .', '.')` discards its result** — official dead code (L878), kept in a comment; the real cleanup is the later `new_ts.replace(' .', '.')`.
- **The declarative replacement is "the last occurrence"**: reverse-find splicing ≡ `rfind`; the `"" in non-empty` semantics makes an empty declarative fall into the else branch and reach the same fallback (m-decl-empty was removed as equivalent).
- **Scoring's bidirectional try-float**: verdicts and expected values convert independently — "4.0"==4.0 hits, "True" never hits a numeric expectation (the numeric verifier, demo_B = create_verifier_demo_text, not the _TF variant, outputs numbers).
- **The flat pool a_idx*3+d**: three fixed slots per candidate (fewer than 3 subquestions leave dead tail slots) — mutating to a_idx*2+d shifts everything.

## Test design (9 cases)

- `candidate_collection`: normalization/dedup ("20"/"5.0"/"5"/empty/non-numeric) + the single-candidate path (scores={0:1}, no subquestions) + a single wrong candidate;
- `subquestion_blank`: blank-structure assertions + numeric-verifier scoring (["20"]*10 full marks / ["7"] zero) + the flat-pool dead slot + oracle comparison;
- `top3_and_numbers`: 4 numbers keep 3, the `"3."` trailing-dot extraction, and the period-spanning `-2.5` that never blanks (the official trap asserted head-on);
- `multi_segment_replace`: the same number blanked in both segments (X count==2) + duplicate numbers producing identical subquestions + `"6."` not replacing;
- `declarative_paths`: cache containing answers[0] replaces the last occurrence / missing goes fallback / empty cache goes fallback + oracle comparison;
- `scoring_and_argmax`: ties to the earliest (both directions) + mixed verdict strings (float/string dual track) + a missing verdict pool scoring 0;
- `comma_and_tail`: thousands commas become periods extracting `"3.000"` + `'..'` cleanup;
- `purity`; `args_validation` (three layers of question/sampling/verdict-pool typing + GT/declarative types + messages).

## Mutation coverage

32 designed → **29 CAUGHT + 3 equivalence removals** (m-empty-keep: the sentinel is the empty string; m-decl-empty: an empty declarative reaches the same fallback through the else branch; m-val-vo / m-msg-vo: the top-level check is backstopped by the per-element check and the top-level message is unreachable).
Coverage: normalization dropped/integerized/dedup dropped, declarative empty-cache/not-containing/find direction/offset, blank token/first-segment-only/dot cleanup/ts slice shift/top-3 widened/regex tightened, scoring expected not converted/verdict not converted/count halved/flat index shifted, argmax tie reversed/single-candidate zeroed/single-candidate route inverted/is_true against the raw string/chosen flipped, seven validation mutations + one message mutation.

## Engineering remarks

- **Run the official semantics before writing fixture assertions**: period-spanning non-blanking, the `"6."` trailing-dot extraction, and the numeric (not TF) verifier — all three were debug-prints first and assertions corrected after; "no blanking happens" is more subtle than "the wrong span replaced".
- **The flat-pool index layout**: 2 candidates × 2 subquestions occupy indices 0,1,3,4 (index 2 is a dead slot) — the first draft laid v consecutively over 0-3, leaving candidate 1 with an empty pool and a false-failing scores {0:9,1:0}.
- `str(float(x))` normalization and the np.argmax tie semantics are the two narrow conventions of main_verifier, locked down by m-norm-* and m-argmax-tie respectively.
