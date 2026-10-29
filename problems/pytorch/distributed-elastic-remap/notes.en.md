## Recovery prerequisite and scope

Once unique in-memory state on a failed machine is lost, rank IDs cannot reconstruct it. This problem explicitly assumes a **consistent durable checkpoint covering every failed shard**. Survivors retain their own shards at that same checkpoint state, and lost shards are read from the checkpoint, not transferred out of failed RAM. The function only produces a restoration plan; it performs no I/O and does not verify an actual checkpoint's existence.

This is the problem's original-shard retention and restoration model, not torchelastic's real membership reorganization or rank assignment implementation. It neither accounts for a full training restart nor repartitions all parameters for the new world size, and it makes no guarantee about actual ZeRO/FSDP recovery workflows. Checkpoint read latency, metadata, optimizer state, restart overhead, and reloading survivor shards are excluded from the byte ledger.

## Deriving the survivor map

Convert the failure list to a set \(F\), so duplicate IDs represent one failure. Every ID must belong to the original rank range. Traverse old IDs in ascending order, retain those outside \(F\), and assign consecutive new IDs. This preserves survivor order and covers exactly \([0,W')\), where \(W'=W-|F|\).

For any surviving old rank \(r\), an independent formulation is

\[
\text{new}(r)=r-|\{f\in F:f<r\}|.
\]

Failed entries are -1. This explains why old IDs cannot simply be reused and provides a per-old-rank test invariant without duplicating the reference loop.

## Deriving checkpoint destinations and balance

After sorting \(F\), assign the \(i\)-th checkpoint shard to new rank \(i\bmod W'\). The modulo operand is the **sorted position**, not the failed old ID. The destination list is always aligned with sorted unique failed IDs, regardless of input order.

Write \(|F|=aW'+b\), with \(0\le b<W'\). The first \(b\) new ranks each receive \(a+1\) extra shards; all others receive \(a\). Extra shard counts therefore differ by at most one across all survivors. Include survivors receiving zero extra shards in this comparison. One survivor receives every failed checkpoint shard; with no failures, every survivor has zero extra load.

## Byte accounting and constraints

Each original shard has size \(s=\text{param_bytes}/W\). Restoring only \(|F|\) failed shards gives `migration_bytes` equal to \(|F|s\). The divisor is the **original**, not the new, world size. Retained and restored data obey

\[
W's+\text{migration bytes}=\text{param_bytes}.
\]

Total parameter bytes may be zero: the rank map and destinations remain unchanged, but the payload is zero. The starter gives the complete types and ranges: a positive integer world size, nonnegative integer bytes divisible by world size, and a list of valid integer failed IDs, excluding bool throughout. Invalid types/ranges, non-divisible bytes, or all ranks failing after deduplication raise `ValueError`; non-divisible byte counts must not be silently truncated. Even with a checkpoint, all-rank failure is outside this model because there is no surviving recipient.

## Complexity and independent verification

Let \(K\) be the failure-list length including duplicates, and \(F\) the number of distinct failed IDs. Validation and deduplication take \(O(K)\), mapping takes \(O(W)\), and sorting takes \(O(F\log F)\). Total time is \(O(W+K+F\log F)\), with \(O(W+F)\) space including returned lists.

Tests exhaust all failure subsets retaining at least one survivor for small world sizes. They verify the mapping per old rank and, from each new rank's perspective, check that it receives every \(W'\)-th entry of the sorted checkpoint sequence. They also verify byte conservation, extra load across every survivor including idle ones, and unsorted/duplicate inputs. These invariants validate checkpoint assignments without merely copying the reference modulo list expression.
