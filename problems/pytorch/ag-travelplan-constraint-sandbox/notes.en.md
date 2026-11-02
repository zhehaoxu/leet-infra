# notes (ag-travelplan-constraint-sandbox)

## Source

Upstream `OSU-NLP-Group/TravelPlanner`; three files were read through the GitHub contents API with
`Accept: application/vnd.github.raw` and checked with `wc -l`:

- `evaluation/commonsense_constraint.py` (547 lines)
  - `evaluation` (L521-531): the **registration order** of the eight checks, which is also the order of
    `checks` and `failed` in the returned dict.
  - `is_reasonable_visiting_city` (L128-160): L133 truncates with `min(days, len(tested_data))`; L136 decides
    whether to split a day into two cities via `'from' in city_value`; L140 rejects when the first day does not
    start at `org`; L148 checks the closed circle; L155 requires every city in `city_state_map`; L157 gates the
    state test on `idx not in [0, len-1] and days > 3`.
  - `is_valid_city_sequence` (L85-124): L98 returns False below 3 nodes; L109 rejects a repeated city unless it
    sits at either end; L119 `count == 1 and 0 < i - 1 < len - 1` — `i` is the **next** run start, so `i - 1`
    is the current run start.
  - `is_valid_restaurants` (L163-194): the three meals share one list and stop at the first repeat.
  - `is_valid_attractions` (L196-213): L204 `unit['attraction'].split(';')[:-1]`, dropping the last segment.
  - `is_valid_transportation` (L215-234): L217 requires day-one transportation (and appends it twice); L231
    has exactly two conflicting pairs — `(Self-driving, Flight)` and `(Taxi, Self-driving)`.
  - `transportation_match` (L57-66): short-circuits taxi, then self-driving, then flight; unknown maps to
    `None`.
  - `is_valid_information_in_current_city` (L236-320): transportation is checked per city, so both cities must
    appear; meals only need `any`; attractions are checked one by one; accommodation is checked against
    `final_city_list[-1]` only.
  - `is_valid_information_in_sandbox` (L323-398): L330-332 take origin/destination from the transportation
    string and fall back to `current_city`; L340 parses the flight number with
    `value.split('Flight Number: ')[1].split(',')[0]` and matches both endpoints; L353/356 query the distance
    matrix and treat `cost == None` as invalid.
  - `is_valid_accommodaton` (L401-424): L411 `count_consecutive_values(data)`; L418
    `len(...) == 1 and unit[1] < minimum nights` — the comparison only happens when **exactly one** row matches.
  - `is_valid_days` (L454-464): the placeholder string
    `You don't need to fill in the information for this or later days.` does not count as a day.
  - `is_valid_visiting_city_number` (L426-452): L447 discards `org` before comparing the set size.
  - `is_not_absent` (L466-518): L470/473 short-circuit on `is_valid_days` and `is_valid_visiting_city_number`;
    L497-507 are the four per-day rules; L510-515 compute the fill rate
    `total_valid_info * 1.0 / (6 * days) < 0.5` where the numerator walks `for key in unit`, so
    **`current_city` counts too**.
  - L24-25: `city_state_map` comes from `database/background/citySet_with_states.txt`, split on `\t`.
- `utils/func.py` (192 lines)
  - `get_valid_name_city` (L127-135): `(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$` splits the name at the **last**
    comma, then strips the parenthetical from the city and strips whitespace; on failure it returns
    `("-", "-")`.
  - `extract_before_parenthesis` (L172-174): `^(.*?)\([^)]*\)`, which only strips a **balanced** parenthetical.
  - `count_consecutive_values` (L176-192): same code as L37-54 of the checker.
- `evaluation/hard_constraint.py` (266 lines): the other five checks in `evaluation` (L245-252)
  (cuisine / house rule / transportation / room type / cost), plus `get_total_cost` (L71-128),
  `is_valid_cuisine` (L158-203) and `is_valid_room_type` (L221-242). **Not used here** — see below.

## Normalization

- **Database to explicit dicts**: the five module-level singletons
  (`Flights()/Accommodations()/Restaurants()/Attractions()/GoogleDistanceMatrix()`) are pandas DataFrames plus
  `os.chdir`. They become the five list/dict tables in `sandbox`, but the semantics are kept verbatim:
  `Name.astype(str).str.contains(...)` stays a **substring** test while `City == city` stays an **equality**
  test. That asymmetry is one of the judged points.
- **City-to-state table becomes `city_state_map`**: upstream reads a text file, here the caller passes a dict.
- **The hard constraints are dropped entirely**: the five checks in `hard_constraint.py` are either a cost
  account over flight and distance pricing (`people_number * price`, `ceil(people / 5)` for self-driving,
  `ceil(people / max_occupancy)` for accommodation) or preference matching (cuisine substring, house rule,
  room type). They are orthogonal to the text-consistency and sequence-structure checks here, and mixing them
  would turn the problem into arithmetic, so only the eight commonsense checks are kept.
- **pandas / numpy / tqdm removed**: `convert_bool_values` (L36-51) exists only to make `np.bool_` JSON-safe;
  this problem returns plain bools.
- **Messages kept verbatim**: including the upstream typos `minumum` and `attaction`, because the messages are
  part of the observable behaviour and pin down which check fires first.

## Mutations

39 mutations, 38 CAUGHT, 1 proven equivalent:

```
m-regex-anchored         anchor the from-to regex with ^      -> base trip catches it
m-lookahead              lookahead reduced to (?=[,])         -> base trip (trailing B fails to match)
m-from-to-nomatch        return ("","") instead of None       -> unparsable_current_city
m-paren-anchored         drop the closing-paren requirement   -> unbalanced_parenthesis
m-name-greedy            (.*?) -> (.+)                        -> empty_name (empty name stops matching)
m-name-no-strip          skip the strip on name and city      -> name_whitespace
m-transport-order        self-driving matched before taxi     -> taxi_wins
m-transport-no-none      return "Flight" when unknown         -> unknown_kind
m-city-pair-space        'from' -> 'from '                    -> unparsable_current_city
m-runs-init              run counter starts at 0              -> min_nights / base
m-seq-len                threshold 3 -> 2                     -> one_day_trip ("from A to A")
m-seq-singleton          singleton boundary <= len-1          -> base trip [A,B,B,B,A]
m-seq-first-repeat       drop the `i != 0` clause             -> equivalent (below)
m-closed-circle          closed-circle test forced off        -> closed_circle
m-days-gate              days > 3 -> >= 3                     -> state_gate_only_above_three
m-unknown-city           city existence forced off            -> city_must_be_known
m-place-substring        name match turned into equality      -> place_substring (R1 vs R10)
m-place-city             city match turned into substring     -> place_city_must_equal (AB vs A)
m-restaurant-seen        dedupe looks at seen[:-1] only       -> restaurant_repeat_same_day
m-attraction-split       split(';') keeps the last segment    -> attraction_semicolon
m-min-nights-rows        len(rows) == 1 -> >= 1               -> min_nights_ambiguous
m-min-nights-eq          nights < min -> <=                   -> min_nights
m-transport-pair         conflict pair taxi/self-driving lost -> transport_conflicts
m-day1-skip              day one only checked for non-empty  -> transport_day1
m-city-consistency-any   any -> all                           -> city_consistency
m-accommodation-city     cities[-1] -> cities[0]              -> city_consistency
m-flight-origin          flight matched on destination only   -> flight_both_cities
m-flight-split           split loses one space                -> flight_lookup
m-distance-key           distance key drops the mode          -> distance_missing
m-needed                 6 * days -> 7 * days                 -> fill_ratio_boundary
m-ratio                  < 0.5 -> <= 0.5                     -> fill_ratio_boundary (exactly 9/18)
m-last-day               off-by-one on the final day          -> not_absent_day_rules
m-meal-space             'from ' -> 'from'                    -> meal_rule_with_newline
m-days-placeholder       placeholder no longer excluded       -> placeholder_day
m-discard-org            org no longer discarded              -> city_number_discards_org
m-bool-int               int check accepts bool               -> input_validation
m-window                 no truncation by days                -> extra_days_are_ignored
m-city-number-org        first-day baseline org -> dest       -> base trip (city_number and not_absent)
m-field-missing          empty string treated as missing key  -> empty_string
```

**Proof of the equivalent mutation**: the `i != 0` clause in
`if city in visited and i != 0 and i != len(cities) - 1` is dead code — `i` only ever increases, and on the
first iteration `visited` is necessarily empty, so "start index 0 with an already visited city" is
unreachable. Exhaustively checked over the alphabet `{A,B,C}` for every sequence of length 1–7 (3279
sequences): with and without the clause the result is **identical** (0 differences). Deleting the clause still
passes every test, so this is not a coverage gap.

## Pitfalls

- **The node count is 2N-1, not N**: `from A to B` contributes two nodes, so the three-day base trip gives
  `[A,B,B,B,A]`. The first draft assumed three nodes and got both the closure and the singleton rule wrong.
- **A one-day trip can never pass the sequence check**: `len < 3` rejects it, and the two-node spelling
  `from A to A` is rejected too. Do not write a test expecting "1 day, 1 city" to satisfy
  `is_reasonable_visiting_city`.
- **The fill-rate numerator and denominator have different sources**: the denominator is `6 * days` while the
  numerator walks **every key** of the day dict — upstream units carry 7 keys including `current_city`, so
  blanking only the three meals still gives 11/18 = 0.61 and passes. The first draft asserted that blanking the
  meals would drop below 50%; it does not, and attractions have to be blanked as well to reach 8/18.
- **Exactly 9/18 = 0.5 passes**: only `< 0.5` fails. That boundary pins both the `<` operator and the factor 6,
  and it is what catches the `m-ratio` and `m-needed` mutants.
- **An attraction string without a semicolon silently disappears**: `split(';')[:-1]` drops the tail, so
  `"Y, B"` takes part in none of the repeat, city-consistency or sandbox checks and nothing is reported.
- **`minimum_nights` is only compared when exactly one row matches**: with two same-name same-city rows the
  whole check is skipped, so test data must keep the table unambiguous.
- **An empty parsed name matches everything**: `", A"` yields name `""`, and `"" in any table name` is true.
  That is a consequence of upstream's pandas `.str.contains`, kept as-is and called out in the hints.
- The three upstream restaurant messages were merged into one format
  (`The restaurant in day %d %s %s is repeated.`), because the assertions only need "which check failed" plus a
  readable message, not three distinct phrasings.

## Deduplication

- `ag-expel-insight-ledger` (752) and `ag-ace-skillbook-ledger` (750) are also "LLM emits text operations, we
  fold them into a ledger". This problem keeps no ledger at all: it only validates and returns a per-check
  `(ok, msg)`.
- Nothing shared with the `context-compression` or `retrieval` collections: no retrieval, no scoring, no
  summarisation here.
- The `eval-*` problems (including `eval-judge-score-parse` (746) and `eval-judge-swap-consistency` (684))
  judge the parsing of free-form model output and win-rate aggregation; this problem judges **structured
  constraints** (city sequence, de-duplication, sandbox existence, fill rate). No overlap.
- The TravelPlanner hard constraints (cuisine / house rule / room type / budget) were **not** turned into a
  problem: they are either `people_number * unit price` arithmetic or preference substring matching, and they
  would require two more pricing tables. Only the commonsense half is kept here.
