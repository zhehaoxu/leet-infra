# notes — fw-serialize-weights-only (order 712)

## Provenance

- Source: the judge environment's own `.venv-eval` `torch/serialization.py` and `torch/_weights_only_unpickler.py` (v2.14.1). The allowlist table was reverse-engineered from `_get_allowed_globals` plus actual protocol-2 serialization output, and closed-loop verified per scenario by the oracle calling the real `torch.load(weights_only=True)` — the table matching official behavior is the anchor.
- Key facts: torch.save writes a zip container; the pickle member lives at `archive/data.pkl` with default protocol 2 — GLOBAL is a two-line text instruction (`module\nname\n`), rendered by genops as `"collections OrderedDict"`. Pure primitive containers emit no GLOBAL.
- Boundary with `fw-resume-checkpoint`/`fw-state-dict-remap`: this is the weights_only allowlist decision layer plus the load_state_dict key-diff ledger, not key-remapping business logic.

## Judge surface

- solve() takes no arguments, 17 fixed scenarios; the oracle uses the real `torch.load(weights_only=True)` for verdicts, value roundtrips (torch.equal) for ok scenarios, real `Module.load_state_dict` for missing/unexpected, and the 2.6 default ledger (bare load of nn.Linear rejected; bare load of a pure tensor dict passes).
- Red-line check: scans the solution source for the call forms `torch.load(` / `.load_state_dict(` (skipped when the harness module has no `__file__`; a loose substring would hit docstring text and caused a false CAUGHT earlier).

## Anchors (all measured against the real torch)

- Allowed group: tensor dict (OrderedDict+FloatStorage+_rebuild_tensor_v2), nested primitives (zero GLOBALs), OrderedDict, Counter, Size, device, dtype (torch float32), set/bytearray (+_codecs encode)/complex, Parameter (_rebuild_parameter, no standalone GLOBAL).
- Rejected group: nn.Linear (torch.nn.modules.linear Linear), SimpleNamespace (types SimpleNamespace), **range → `__builtin__ xrange`** (protocol-2 mapping quirk), numpy (dtype/ndarray/_reconstruct).
- lambda: PicklingError at save time → save_fail classification (not a load rejection).
- load_state_dict: `0.bias < 0.weight` lexicographic sorted; foreign ledger missing=["0.bias","0.weight"] / unexpected=["bias","weight"]; +extra.weight → unexpected=["bias","extra.weight","weight"]; strict RuntimeError first-line prefix; root-module keys carry no attribute-name prefix.

## Mutation log (11/12 caught)

linear-allow, xrange-allow, np-allow, dtype-drop, counter-drop, storage-drop, verdict-flip, scan-wrong-member (wrong zip member), missing-unexpected-swap, savefail-to-ok, sort-drop (dropping sorted — the three unexpected keys' insertion order ≠ lexicographic, covered by the scenario).

Two equivalent mutations recorded: dropping the STACK_GLOBAL scan branch (protocol 2 emits none — dead code; verified across all 17 scenarios) and strict-flip (an accidentally tautological mutation). The STACK_GLOBAL branch is kept to mirror official `find_class`'s protocol-4 intent.

## Red lines

No torch.load/load_state_dict in the solution (hand-written decision layer); no safe_globals dynamic registration (_add_safe_globals), torch.package, jit.save; no distributed checkpoint (dcp); no overlap with `fw-resume-checkpoint` (key-remapping business).
