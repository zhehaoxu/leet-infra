# Zep Bi-temporal Edge Invalidation Ledger

Zep's temporal knowledge graph (arXiv 2501.13956, implemented as
`getzep/graphiti`) keeps **two pairs** of timestamps on every fact edge:

- `valid_at` / `invalid_at` is **event time** — when the fact became true in the
  world, and when it stopped being true;
- `expired_at` is **transaction time** — when the system decided the record was
  no longer true.

The two must not be conflated: when "Alice moved to Berlin in 2024-03" supersedes
"Alice lives in Paris", the old edge gets `invalid_at = 2024-03` (when it stopped
being true) and `expired_at = the ingest moment` (when we found out).

This problem strips one ingest down to a pure function: the LLM's dedupe /
contradiction call becomes the `decisions` argument, timestamp extraction becomes
`timestamps`, the transaction moment becomes `now`, and everything left is
**index translation, dedupe, and invalidation**. Implement

```
solve(extracted, related_edges=(), existing_edges=(), decisions=None,
      now=None, timestamps=None)
```

## Inputs

- `extracted`: the newly extracted edge
  `{"uuid", "fact", "source", "target", "valid_at", "invalid_at"}`.
- `related_edges`: existing edges for the same node pair. Used both as **dedupe**
  candidates and as invalidation candidates.
- `existing_edges`: a wider invalidation candidate pool. Used for invalidation
  **only**, never for dedupe.
- `decisions`: the LLM response
  `{"duplicate_facts": [int], "contradicted_facts": [int]}`. The two index ranges
  are **continuous**: `0..len(related_edges)-1` addresses `related_edges`,
  `len(related_edges)..len(related_edges)+len(existing_edges)-1` addresses
  `existing_edges`.
- `now`: the transaction timestamp of this ingest (required, UTC ISO string).
- `timestamps`: the timestamp-extraction result `{"valid_at", "invalid_at"}`.
  Optional.

UUIDs are unique across both edge lists and never collide with `extracted`.

## Output

```
{
  "path": "empty" | "fast" | "resolve",
  "resolved": {"uuid", "fact", "valid_at", "invalid_at", "expired_at"},
  "resolved_is_new": bool,
  "duplicate_ids": [int],
  "duplicate_uuids": [str],
  "invalid_duplicate_ids": [int],
  "invalid_contradiction_ids": [int],
  "candidate_uuids": [str],
  "sorted_candidate_uuids": [str],
  "invalidated": [{"uuid", "valid_at", "invalid_at", "expired_at"}],
  "edges": {uuid: {"uuid", "fact", "valid_at", "invalid_at", "expired_at"}},
  "writes": [{"uuid", "field", "value"}]
}
```

`edges` snapshots **every edge touched** in this call (the new edge plus both
candidate lists) after the call; `invalidated` holds the subset that was judged
invalid, in traversal order. All timestamps are emitted as UTC ISO strings
(`...Z`); missing values are `null`.

`writes` is the audit stream: one entry per write, `field` being
`"valid_at"` / `"invalid_at"` / `"expired_at"` and `value` the stored value.

## Three paths

1. **`empty`**: both candidate lists are empty. There is nothing to compare
   against — **not even the fast path runs**. The new edge just gets its
   timestamps, and no invalidation happens.
2. **`fast`**: some edge in `related_edges` has a `fact` that is **character-for-
   character identical** after normalization *and* the same `source` and `target`.
   Reuse that edge: `resolved` points at it, `duplicate_ids` is empty (not `[i]`),
   `decisions` is never read, and nothing is invalidated. Normalization = lowercase
   → collapse runs of whitespace to one space → strip. The fast path scans
   **`related_edges` only**.
3. **`resolve`**: everything else. Translate the indices, then dedupe and
   invalidate.

## Dedupe

- Out-of-range `duplicate_facts` indices (`< 0` or `>= len(related_edges)`) do not
  raise; they are recorded verbatim in `invalid_duplicate_ids`. Valid ones keep
  their **original order** and may repeat.
- `resolved` is the edge addressed by the **first** valid duplicate index; with
  none, `resolved` is the new edge.
- `duplicate_uuids` mirrors `duplicate_ids`, repetitions included.
- `timestamps` is applied **only when `resolved` is the new edge**. When the new
  fact merges into an old edge, that edge keeps its own timestamps.
- Before applying, one gate: if `resolved` already has **either** `valid_at` or
  `invalid_at`, the whole extraction is skipped. Each field is then written only
  when its value is a non-empty string, `valid_at` first.

## Invalidation

Out-of-range `contradicted_facts` indices (`< 0` or
`> len(related_edges) + len(existing_edges) - 1`) are recorded in
`invalid_contradiction_ids`. Valid ones are translated into candidates
(`related_edges` segment first, then `existing_edges`), **preserving order and
repetitions** — that is `candidate_uuids`.

Then three steps:

1. **The new edge already says it is history**: if `resolved.invalid_at` is set
   and `expired_at` is empty, stamp `expired_at = now`.
2. **A later candidate supersedes the new edge**: only while
   `resolved.expired_at` is still empty, sort the candidates by
   `(valid_at is None, valid_at)` ascending and stable (missing `valid_at` goes
   **last**), then find the first candidate whose `valid_at` is **strictly
   greater** than `resolved.valid_at`. On a hit, set
   `resolved.invalid_at` to that `valid_at`, stamp `expired_at = now`, and
   **stop**. This step rewrites the candidate order in place, so step 3 and
   `sorted_candidate_uuids` both use the sorted sequence; when step 1 already
   stamped `expired_at`, **no sorting happens** and both keep the original order.
3. **Judge each candidate**. For every candidate in the sorted sequence
   (duplicates re-enter, but the first rule below screens them out):
   - the candidate is already dead and died at or before `resolved.valid_at`
     → skip (**inclusive**);
   - the new edge itself died at or before the candidate's `valid_at` → skip
     (**inclusive**: an edge that was already gone has no standing to supersede);
   - otherwise, if the candidate's `valid_at` is **strictly less** than
     `resolved.valid_at`, invalidate it: `invalid_at = resolved.valid_at`, and
     `expired_at = now` **only if it was empty** — an existing value is kept.

A tie (candidate `valid_at` exactly equal to `resolved.valid_at`) invalidates
nothing.

## Example

```python
extracted = {"uuid": "new", "fact": "Alice lives in Berlin",
             "source": "alice", "target": "berlin",
             "valid_at": "2024-03-01T00:00:00Z", "invalid_at": None}
related_edges = [{"uuid": "old", "fact": "Alice lives in Paris",
                  "source": "alice", "target": "berlin",
                  "valid_at": "2024-01-01T00:00:00Z", "invalid_at": None,
                  "expired_at": None}]
decisions = {"duplicate_facts": [], "contradicted_facts": [0]}
now = "2024-06-01T00:00:00Z"

returns: path="resolve", resolved.valid_at="2024-03-01T00:00:00Z",
         invalidated=[{"uuid": "old", "valid_at": "2024-01-01T00:00:00Z",
                       "invalid_at": "2024-03-01T00:00:00Z",
                       "expired_at": "2024-06-01T00:00:00Z"}]
```

## Constraints

- Pure function: no network, no file I/O, no mutation of the inputs.
- A missing `now`, or missing `decisions` while candidate edges exist, raises
  `ValueError`.
- Timestamps are ISO strings (`Z` or `+00:00`); naive values are read as UTC.
