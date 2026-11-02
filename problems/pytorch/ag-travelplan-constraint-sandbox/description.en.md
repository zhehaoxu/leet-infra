# TravelPlanner Constraint Sandbox: Eight Commonsense Checks Over an Itinerary

TravelPlanner (arXiv 2402.01622) neither trains a model nor scores one: the LLM emits a day-by-day itinerary
JSON and a sandbox runs two **hand-written checkers** over it — `hard_constraint.py` for preferences and
budget, `commonsense_constraint.py` for "does this trip make sense at all". This problem is the second one:
eight pure functions plus a fill-rate account. In `OSU-NLP-Group/TravelPlanner` it is a script (global pandas
tables, `os.chdir`); here the tables, the city-to-state map and the distance matrix are explicit arguments.

## Interface

```python
def solve(query, itinerary, city_state_map, sandbox) -> dict
```

- `query`: `{org, dest, days, visiting_city_number, ...}` — origin city, destination state, trip length and
  the number of distinct cities to visit.
- `itinerary`: one dict per day with
  `{current_city, transportation, breakfast, lunch, dinner, attraction, accommodation}`;
  `current_city` is often written `from A to B` and attractions as `"X, A;Y, A;"` (**semicolon-terminated**).
- `city_state_map`: `{city: state}`, the upstream `citySet_with_states.txt` table.
- `sandbox`: five tables — `restaurants`, `accommodations`, `attractions`, `flights`, `distances`.

Returns `{"checks": {name: {"ok", "msg"}}, "failed": [...], "passed": bool}`. The eight checks follow the
order of the upstream `evaluation()` and the failure list keeps the same order.

## Input / Output

```
Input: query  = {org:"A", dest:"S1", days:3, visiting_city_number:1}
       day[0] = current_city "from A to B", transportation "Self-driving, from A to B",
                breakfast "R1, A", lunch "R2, A", dinner "R3, A",
                attraction "X, A;", accommodation "H2, B"
       day[1] = current_city "B", transportation "-", breakfast "R4, B", lunch "R5, B",
                dinner "R6, B", attraction "Y, B;", accommodation "H2, B"
       day[2] = current_city "from B to A", transportation "Self-driving, from B to A",
                breakfast "R7, B", lunch "R8, B", dinner "R9, B",
                attraction "V, B;", accommodation "-"
Output: passed = True
        failed = []
        city nodes = [A,B,B,B,A]

Input: day[0].dinner = "R1, A"
Output: failed = ["is_valid_restaurants"]
        checks["is_valid_restaurants"] = {"ok": False,
            "msg": "The restaurant in day 1 dinner R1, A is repeated."}
```

## Rules

- **City nodes**: when `current_city` contains the substring `from`, split it with
  `from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)` into two cities, otherwise take one single city; every value
  then goes through `extract_before_parenthesis` (`^(.*?)\([^)]*\)`, which only strips a **balanced**
  parenthetical). An unparsable `from` raises `ValueError`. A three-day trip therefore yields five nodes.
- **First day and closure**: on day 0 the first node must equal `query['org']`; the first and last nodes must
  be the same city.
- **Sequence validity**: fewer than 3 nodes is invalid; after collapsing the list into consecutive runs, a run
  that starts at a city **already seen** and is not at either end is invalid, and a run of length one in the
  middle is invalid. Every node must exist in `city_state_map`, and when `days > 3` each **interior** node must
  belong to `query['dest']`.
- **Restaurants**: the three meals share one de-duplication table keyed by the **whole string** (name plus
  city), across all days; two meals of the same day hitting the same entry also count as a repeat.
- **Attractions**: `attraction.split(";")[:-1]` — the last segment is dropped, so an attraction string
  **without a trailing semicolon counts as empty**; the rest are de-duplicated across days.
- **Accommodation**: the `accommodation` **key** must exist (a missing key reports
  `No Accommodation Info.`, while `-` does not); the values are collapsed into runs with
  `count_consecutive_values` and each run length is compared against the table's `minimum_nights`. The compare
  only happens when name plus city matches **exactly one** row; several matching rows skip the check.
- **Transportation**: day one must be present and not `-`; the kind is resolved by substring in the order
  `taxi → self-driving → flight` (case-insensitive, unknown maps to `None`); only two combinations conflict —
  self-driving with flight, and taxi with self-driving.
- **City consistency**: every field of the day has to mention a city of that day (substring match) —
  transportation must mention **both** cities, meals and each attraction only one of them, and accommodation
  only the **last** city of the day.
- **Sandbox lookups**: restaurants, accommodations and attractions match by "name is a substring of the table
  name plus **exact** city", with the name split at the **last** comma
  (`(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$`, city then stripped of its parenthetical and of whitespace); a flight
  number is taken between `Flight Number: ` and the next comma and must match both origin and destination;
  self-driving and taxi endpoints must exist in the distance table.
- **Missing information**: two short-circuits first (`days` not fully covered → `Invalid Days`; the number of
  cities other than the origin not equal to `visiting_city_number` → `Invalid City Number`); then all six field
  keys must exist; then per-day rules (a cross-city day needs transportation, a single-city day needs an
  attraction, a non-final day needs accommodation, a non-cross-city day needs all three meals); finally the fill
  rate — "non-empty and non-`-` keys over `6 × days`" — must be `>= 0.5`, else
  `The absent information is more than 50%.`
- **Validation**: `days` and `visiting_city_number` must be non-`bool` ints with `days >= 1`, `org` and `dest`
  must be strings, `itinerary` must be a non-empty list of dicts carrying a string `current_city`; otherwise
  `ValueError`.

## Hints

- `from ... to ...` splits one day into **two** nodes — the easiest account to get wrong: three days give five
  nodes, not three.
- The sequence check wants every interior city to stay at least two nodes, so a one-day or two-day trip
  (fewer than 3 nodes) can **never** pass it.
- The fill-rate numerator walks **every key** of the day dict, `current_city` included, while the denominator
  is fixed at `6 × days`. A fully filled trip scores above 1, and blanking only the three meals still leaves
  11/18.
- An attraction string that forgot its trailing semicolon does not raise anything — it silently disappears from
  the repeat check, the city-consistency check and the sandbox lookup.
- Table names match by substring, and an empty parsed name (writing `", A"`) makes `'' in any name` true, so
  any restaurant of that city matches.
