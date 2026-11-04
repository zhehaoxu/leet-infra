# Solution Notes: PAB Attention Broadcast Step

## Source Localization

- Source file: diffusers `hooks/pyramid_attention_broadcast.py` (314 lines, PAB paper arXiv 2408.12588). Core: `_apply_pyramid_attention_broadcast_on_attention_class` (three-way matching + priority dispatch, L241-287), `PyramidAttentionBroadcastHook.new_forward` (open interval + modulo gate, L156-174), `apply_pyramid_attention_broadcast` (all-None fallback to spatial=2, L220-230), and the three identifier defaults in `_common.py` (spatial includes broad words like "blocks"/"layers", temporal only "temporal_transformer_blocks", cross overlaps spatial's first two).
- The official hook attaches to every attention module, with `iteration % block_skip_range == 0` deciding recomputation inside `new_forward`; `or self.state.iteration == 0` is fully covered by 0%n==0 (dead code) — dropped from the judgement domain and recorded.
- Dispatch is a one-time hook-creation decision, the gate is a per-step decision; the judgement domain merges them, letting block_type flow out of dispatch as a natural cross-check channel.

## Math Notes

1. **Open interval**: `lo < t < hi` strictly on both sides. t==lo or t==hi is outside — with a cache established, compute is forced. Mutants m-open-lo/m-open-hi catch each side.
2. **Modulo sequence**: iteration starts at 0 and increments on every call (broadcasts included); `iteration % N == 0` puts compute at 0, N, 2N…. The %→!=0 (inverted) and %N==1 (shifted) mutants are both caught by sequence cross-checks.
3. **Cache update semantics**: `state.cache = output` runs unconditionally — on broadcast steps the cache becomes an alias of the old cache (value unchanged), on compute steps it refreshes immediately. m-cache-noedit (cache not updated) is exposed by "out-of-range compute → back in-range broadcast" (old value vs new).
4. **The priority chain**: the spatial → temporal → cross if/elif order plus the `not is_cross`/`is_cross` mutual-exclusion guards. The official default spatial_ids' "blocks" with re.search swallows temporal layer names — official behaviour, not a bug (temporal only applies when spatial_range is None); the priority-swap mutant is caught by 11 cases.

## Implementation Notes

- Validation order: name/is_cross/tensor/current_timestep → three block_skip_ranges (None or ≥1 non-bool int) → three timestep ranges (two-integer sequences) → three identifier groups (string sequences) → state two keys (unknown before missing, iteration non-bool ≥0, cache tensor or None).
- The all-None fallback to spatial=2 must run before dispatch.
- Unmatched layers: `dict(state)` passed through unchanged, output=hidden_states, computed=True, block_type=None.
- The `isinstance(identifiers, str)` check is dead code (a string is not a list/tuple, so the later check raises the same error) — recorded as an equivalent mutation.

## Testing Notes

- Dispatch quadrants: spatial hit + cross excluded, temporal preempted by spatial (official default ids), temporal taking over when spatial_range=None, cross requiring is_cross=True + a hit, unmatched layers → None.
- Gate sequences: N=2 alternating [T,F,T,F,T], N=3 [T,F,F,T,F], with broadcast-step outputs compared value by value (the previous compute's output).
- Both open-interval endpoints t==lo/t==hi always compute; only mid-range values may broadcast.
- cache=None forces compute (hand-built state iteration=5 + cache=None); iteration=0 + non-None cache also computes (0%n==0).
- After an out-of-range compute the cache refreshes immediately (the observation point for m-cache-noedit).
- Independent temporal/cross timestep ranges (temporal (0,50), cross (50,100) not leaking into each other).
- Purity + unmatched-layer state pass-through + a validation battery (skip_range 0/True/-1, range length/dtype, identifiers as string/with non-strings).

## Mutation List (23, all CAUGHT; 1 equivalent dropped)

Dispatch 6: spatial dropping the cross exclusion, cross flag negated, priority swapped, any→all, search→match, default fallback removed. Gate 6: cache-None check dropped, open-interval left <=, right <=, within condition dropped, modulo inverted, modulo shifted. Output/state 5: broadcast output swapped to hidden, compute output swapped to cache, cache not updated, iteration not incremented, unmatched computed flipped. Validation 6: name, is_cross, skip_range lower bound, timestep range length, identifiers, state missing keys + message.

**Equivalent-mutation record (dropped)**: ① the identifiers string pre-check — fully subsumed by the list/tuple check (same error); ② the official `iteration == 0` branch — 0%n==0 always true, not implemented.

Experience: ① thin-gate problems gain mechanism density from a dispatch layer (regex + flags + priority); merging "creation-time decisions" with "per-step decisions" makes block_type a natural cross-check channel; ② official dead code (the 0%n==0 branch, the string pre-check) is more honestly identified and dropped than copied; ③ mutation fragments shared between docstring and code are located by indentation (`or iteration % ...` → `\n        or iteration % ...`).
