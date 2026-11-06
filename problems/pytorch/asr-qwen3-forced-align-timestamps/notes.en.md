# Pitfall notes (order 831)

## Source-reading caveats

- Repo `QwenLM/Qwen3-ASR`, `qwen_asr/inference/qwen3_forced_aligner.py` (483 lines). grep kept returning nothing on this repo's .py files (reason unknown); read it via Read / python re.search section by section.
- **The LIS is non-decreasing (`<=`), not strictly increasing (`<`)**: equal timestamps are normal points. The `<` mutant is caught on `[5,5,5]`'s dp / is_normal (fixed values do not change — the intermediate dp / lis_indices / is_normal must be exposed, or that mutant survives).
- `parent` updates only when `dp[j]+1 > dp[i]` is strictly greater (ties keep the earlier); `max_idx = dp.index(max(dp))` takes the **earliest** tie. The dp-tie case `[10,20,10,20,15]` (dp=[1,2,2,3,3]) pins it: earliest → chain [0,1,3] → fixed [10,20,20,20,20]; latest → chain [0,2,4] → fixed [10,10,10,10,15].
- **Run length <=2 / >2 is the branch point**: <=2 chooses between left/right neighbor values by distance (`(k-(i-1)) <= (j-k)`, ties take left); >2 uses `step=(right-left)/(anomaly_count+1)` interpolation. The denominator is **count+1** (both endpoints included), not count.
- Left/right values are read from **`result` (already repaired)** rather than `data` — but for normal points `result[k] == data[k]`, so this is a true equivalence; no mutant for it (avoids a fake red).
- `int(v)` truncation (positive floats toward zero = floor), **not round**: interpolation producing 12.5 / 17.5 → 12 / 17; `round` gives 12 / 18 — a dedicated case catches it.
- Anomalous values must be **all smaller than the left normal neighbor** to build a length-3 run (otherwise 999/888-style values join the LIS themselves and the run shrinks to 2, taking the wrong branch). The first hand computation used [0,10,999,888,777,20], whose actual normal set is [0,1,2,5] — the interpolation branch never fired. Switched to [0,10,9,8,7,20] to pin count=3 interpolation.
- In `parse`, `round(v/1000.0, 3)`: fixed is already int so `/1000` has exactly 3 decimals — deleting round is a **true equivalence** (dropped); changing precision 3→2 is observable (1.666→1.67).
- `align()` runs `parse_timestamp` (raw int dicts) then `/1000 + round(3)`; the parse mode here merges both steps, but exact-value assertions expose both fixed and spans.
- Japanese/Korean tokenization uses nagisa / soynlp (absent from the judge env); Chinese/English always take `tokenize_space_lang` — the judge covers only the space-lang branch, pinned in the docstring.

## Test-design pitfalls

1. **Hand anchors must trace the LIS chain**: the first version for cases 7/8 computed `dp.index(3)` as 2 (actually 3), making lis_indices and fixed all wrong — one dry run exposed it; recomputed dp layer by layer and backfilled.
2. **`check_masked_timestamps` slot-count assertion**: ids length must equal 2×word count; the first version hard-coded 6 with 2 words (4 slots) → FAIL, changed to `2*len(wl)`.
3. **True-equivalence mutant m-fix-walk survived**: changing `j = i` to `j = i + 1` leaves the while endpoint unchanged, so both `j-i` and `range(i,j)` are unchanged — fully equivalent. Dropped it; replaced with `j += 1` → `j += 2` (run boundary and count both change) to catch it. **Lesson: shifting a loop start does not necessarily change semantics — verify the endpoint first**.
4. The LIS parent mutant (`>` → `>=`) is unobservable on dp (the dp array is unchanged) — caught by the lis_indices assertion `[0,1,3]`. Intermediate-value exposure pays off a second time.
5. Missing-side cases (no left at the start / no right at the end) and distance ties (run length 1 gives `1 <= 1` → left) each get their own case, preventing the "both sides present" branch from masking them.
6. End-to-end consistency: encode's placeholder-token count == masked_timestamps' value count == 2×word count, pinned by a composition assertion.

## mut design (20 mutants all CAUGHT + 1 true equivalence dropped)

- Tokenize family: apostrophe dropped / digits dropped / CJK not split / flush loses token — mixed CJK-Latin + punctuation + empty-segment cases;
- Encode family: single placeholder between words / single tail placeholder / missing audio prefix — exact string assertion + token count;
- NAR family: inverted mask / multiply changed to divide — non-placeholder large values excluded + two segment_time controls;
- LIS family: non-decreasing → strict / parent tie update / max tie to latest — five-way exposure of dp / parent / lis_indices / is_normal / fixed;
- Repair family: threshold 2→1 / distance `<=`→`<` / interpolation denominator losing +1 / int→round / step 1→2 — run-length 1/2/3 case families + the 12.5/17.5 divergence;
- Parse family: start wrong column / /1000→/100 / round 3→2 — exact per-item span assertions.
