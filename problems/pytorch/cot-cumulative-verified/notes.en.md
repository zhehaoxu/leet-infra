# Solution Notes

## Approach

The deterministic core of the official iiis-ai/cumulative-reasoning repo:

1. **Game24/game24-openai.py solve()** (L537-590): the state-graph growth loop — the records/last_step/f/forbidden bookkeeping, the two-phase sampling distribution (L545-553), the four gates (charset, non-whitespace, the Invalid verdict), the verifier three-verdict branching (L562-585), and the parent-chain reconstruction (L569-574);
2. **is_sure / valid_output / valid_judgement** (L38-41, L135): `is_sure = output == 'sure'`, the three verdicts `["sure", "likely", "impossible"]`, and the judgements `["Valid", "Invalid"]`.

Source-verification note: the todos candidate ⑤ described "proposer+verifier cumulative-state ledger — propositions admitted by the verifier, the cumulative state growing monotonically", which matches the official Game24 implementation (the FOLIO variant is a five-gate admission pipeline and the MATH variant is a prompt pipeline without per-proposition verification — both have thin increments and are not filed).

Out of judging scope: the guidance programs and prompt assembly, the OpenAI calls with run()'s MAX_RETRY retries, numpy.random.choice sampling (chosen is an explicit input), the expand merge LLM (its output is an explicit input / the reconstruction payload is returned directly), the `__main__` sympy check, tqdm/pandas.

Boundary with existing problems:
- **rs-toT / rs-lats / ToT family**: tree-search nodes are selected by a scoring function; here the state is admitted by the **verifier's three-verdict semantics** (likely admits / sure terminates / impossible discards) and paths are rebuilt along parent pointers;
- **775/776/777**: demo assembly, sample selection, candidate scoring; this problem is a **cross-round cumulative state graph** — the state is a derived string (the remaining-number combination), not a text candidate;
- **778**: RoT's subquestion sequence derives mechanically from the problem grammar and is compiled in one shot; here each round's proposer output is an explicit input and verdicts drive the branching.

## Key implementation points (official semantics, pinned item by item)

- **Per-round processing order**: chosen → a = first-line extraction → gate A (characters outside `[^0-9+\-*/.(),=\s]`) → gate B (`\S` check, second operand is the **full** stripped remaining) → Invalid → verifier. Gates A/B/Invalid are all pure continues (interchangeable).
- **a's two strips**: strip the full string, take split('\n')[0], strip again — `'6, 8  \njunk'` → `'6, 8  \njunk'` → first line `'6, 8  '` → `'6, 8'`; junk neither enters the state nor affects gate B.
- **likely's five-step admission**: `forbidden[tmp].append(next_step)` (first) → `forbidden[a] = []` → `records.append(a)` → `parent[a] = tmp` → `last_step[a] = next_step`. Removing `forbidden[a] = []` does not surface immediately — it bites when `a` is later **chosen**: `forbidden[a]` raises KeyError and the following rounds are skipped (the chained-admission case).
- **sure's steps structure**: seeded with `[equation first line, next + ' (left a)']`, the while loop walks toward the root via parent, appending `last_step[t] + ' (left t)'` per step, then reverses — root-side steps first, the merged equation last. The official code returns the expand LLM's output; this port returns the steps (the payload joined into expand), with the deviation recorded.
- **Duplicate states are silently dropped**: likely with `a` in records does **nothing** — forbidden[tmp] is not appended (the dup_drop case pins forbidden holding a single entry).
- **Failure marker = trycnt**: not the number of rounds run (running out of rounds early still returns trycnt).
- **Two-phase sampling distribution**: `i < 1/2 * trycnt` is **float division** — with trycnt=5, i=2 is still biased; with n>1, fill(0.5/(n-1)) then overwrite p[0]=0.5; with n=1, p[0]=1.0 directly.
- **Silent skip**: the official try/except wraps the whole loop body; any exception (including a `forbidden[tmp]` KeyError) continues. With chosen as an explicit input, a round whose chosen is not in records is skipped under the same semantics.

## Test design (11 cases)

- `dist`: first/second-half boundaries for even and odd trycnt (i=2 biased, i=3 uniform), n=1, trycnt=0;
- `success_basic`: the Invalid gate + a padded next (stored stripped) + a three-step sure reconstruction + the state snapshot;
- `deep_chain`: two-level ancestor reconstruction (root-side steps first, equation last);
- `gates_no_success`: gate A ('!') / gate B (all whitespace) / two normal admissions / duplicate drop / impossible — the state pins records/parent/last_step/forbidden in full;
- `dup_drop`: a duplicate likely drops the whole round, forbidden keeps a single entry;
- `multiline_edges`: remaining first-line extraction (tail spaces stripped again), next's internal newline stored in last_step, equation's trailing whitespace cut by the first-line strip;
- `forbidden_accumulation`: two admissions from the same parent → two forbidden entries + an impossible round with no action + reconstruction from a grandchild state;
- `success_early`: the first sure returns immediately, later sures do not overwrite the marker;
- `round_boundaries`: rounds>trycnt truncation / rounds<trycnt early finish (marker=trycnt) / chosen not in records silently skipped / trycnt=0;
- `purity`; `args_validation`: mode/puzzle/trycnt/rounds/round-keys/dist parameters + bool exclusion + an in-round bad value (next=None) silently skipped.

## Mutation close-out

44 designed → **41 CAUGHT + 3 equivalence removals** (m-gateA-B-order: the two gates are pure continues, swapping them has no side effects; m-likely-a-strip: the official `a = a.strip()` dead code added back is equivalent; m-eq-nostrip-head: no fixture equation has leading whitespace, so the first strip is redundant).

Coverage: five dist states (float/floor midpoint, root bias, fill value, uniform value, n=1), four gate states (gate A dropped/charset narrowed, gate B dropped/or→and), two judgement states (case, Valid-blocks inverted), three verifier states (sure inverted, likely case), five admission states (dup check both ways, parent to root, last_step value swapped, forbidden append/new-entry removed), seven sure-path states (equation whole string/no tail strip/dropped, wrong left placeholders, no reverse, walk skips to root, always-true while), three extraction states (remaining without first-line split/no tail strip, next unstripped), three loop states (bound by len(rounds), marker by len(rounds), sure without break), eight validation states.

## Engineering remarks

- **Official-sandbox cross-check first**: exec the official solve() source into a stub environment (four stubs: run/choice/args/pbar) and dump every fixture expectation (including state snapshots and the steps payload) from the sandbox — hand-deriving a 4-level state graph always errs. The key stub convention: **the gen call reads rounds[idx] and advances idx; valid/verifier/expand read rounds[idx-1]** (one dict per round; the verdicts belong to the same round as chosen).
- **The official success path returns the expand output, not the steps** — copying that verbatim would leave the reconstruction unobservable in the judging scope (all its mutations equivalent); returning the steps payload is what makes the 7 mutations m-parent-walk/m-steps-noreverse etc. catchable.
- **Why the state mode is necessary**: the four mutations m-gateA-drop, m-dup-check-true, m-forbidden-noappend, and m-bound-len leave the solve-mode result unchanged at (False, N, []); only the bookkeeping snapshot distinguishes them — the observation window is a hard requirement of mutation close-out.
- **Character-class trap in mutation scripts**: adding a **duplicate** character to a negated class ('=' was already in it) is an identity mutation — narrowing the semantics requires **removing** a character.
