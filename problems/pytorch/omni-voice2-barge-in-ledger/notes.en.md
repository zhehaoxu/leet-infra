# Pitfalls and source-verification notes (order 833)

## Source verification

- Repo `AIIT-GLITCH/voice2`, default branch **`master`** (the `main` tree API returns 404); `raw.githubusercontent.com` timed out / 502 repeatedly, so the sources were downloaded with the GitHub API `contents` endpoint plus `base64.b64decode` into `/tmp/voice2_src/`.
- Only the pure-CPU control-plane sources are used: `sounddevice.OutputStream` is stubbed, `logging_util.event` and `time.monotonic` are patched to a controllable clock; no threads, no audio.

```
voice2/enums.py                 8 states / 3 floors / 14 reasons / 4 sources
voice2/state_controller.py      _VALID 28 pairs, STOPPED no-exit, floor block, history
voice2/floor_manager.py         can_agent_speak / request_agent_floor / ...
voice2/interrupt_controller.py  debounce (first trigger wins) + three reactions
voice2/invariants.py            playback_is_allowed priority + INV1 / INV2
voice2/logging_util.py          LatencyTrace duration / to_dict
voice2/workers/playback.py      stale guard, chunk split, _drain on interrupt
voice2/config.py                chunk_ms=20 / sample_rate=22050 / debounce_ms=200
```

- The official `playback.py` uses `from ..enums import`, so it must live in a `voice2/workers/` subpackage; placing it at the package root raises `attempted relative import beyond top-level package`.
- The official `start_new_turn()` increments `current_turn_id` before `snapshot` reads it, so the snapshot comparison must pass the incremented value as `current_turn_id`.
- The comparison script `/tmp/verify833.py` runs with `.venv-eval/bin/python` (`.venv-backend` has no numpy): transition 192 + turn 4 + floor 360 + interrupt 270 + invariants 192 + latency 6 + playback/_speak 108 cases, plus `chunk_samples / _split / is_stale` — **1132 cases, all identical**.

## Mechanism details

- **`_VALID` has 28 entries, not 29**: the first draft was off by one; the oracle assertion `len == 28` exposed it, which also fixed `n_ok == 83` (28 pairs × 3 floors minus the 1 case of `USER` owning the floor while entering `SPEAKING`, which is blocked).
- **`STOPPED -> STOPPED` returns `invalid_transition`**: the no-exit condition only blocks transitions with `new_state != STOPPED`, while `STOPPED -> STOPPED` is itself absent from `_VALID`; the first hand-computed anchor wrongly expected acceptance and judge reported 5/7.
- **The three reactions cannot be rolled back**: `InterruptController.trigger` first sets `interrupted`, then `handle_interrupt` moves the floor to `USER`, and only then calls `transition(INTERRUPTING)`. Even if the transition is rejected by `VALID` (e.g. interrupting from `IDLE`), `interrupted` and `floor` have already changed; only `state` and `history_len` stay unchanged. All three must be asserted separately.
- **`request_agent_floor` calls `can_agent_speak()` without a `turn_id`** (official default 0): this is the easiest detail to get wrong, and a dedicated mutant `request-agent-turn-id` catches "passing an extra turn_id".
- **`playback_is_allowed` five-level priority**: `shutdown > interrupted > stale_turn > user_owns_floor > ok`; the stale text is `"stale_turn:%d!=%d" % (turn_id, current)` and the order must not be reversed.
- **INV1 forces a repair, INV2 only audits**: when SPEAKING with a floor other than AGENT, the floor is forced to AGENT with `repaired=True`; when interrupted while SPEAKING, only a violation string is appended and the state is untouched.
- **Double rounding in the latency ledger**: `LatencyTrace.to_dict` first does `round(diff, 4)` for spans, then `int((d or 0) * 1000)` for the key durations; there are separate mutants for `round(..., 3)` and for rounding the millisecond value.
- **Discard-ledger semantics**: `queued_chunks_discarded = qsize()` is the **whole queue length** (including other turns), while `_drain` discards only items with `item[1] == turn_id` and keeps the rest; `drained / kept / remaining_queue` must all be exposed, otherwise a "discard everything" mutant survives behind the `qsize` figure.
- `chunk_samples = int(sample_rate * chunk_ms / 1000)`, default `22050 * 20 / 1000 = 441`.
- Floating-point trap: with `now=0.15, last=0.0, debounce_ms=150`, `int(0.15 * 1000) == 149` because of binary representation; the hand anchor uses exactly representable `now=0.25, last=0.0, debounce_ms=400`, expecting `since_ms=250`.

## Test design

- 7 `check_*` functions: `transition / turn / floor / interrupt / invariants / latency / playback`; judge reports `7/7 ACCEPTED` in both `run` and `submit` modes.
- Full-grid comparisons: `transition` over 8 states × 8 targets × 3 floors; `floor` over 5 actions × 8 states × 3 floors × 3 turns; `interrupt` covering debounce 100/200/500, the `since == debounce` boundary, and a trigger from `IDLE` whose transition is rejected; `invariants` over the five-level priority full grid plus INV1/INV2; `playback` over stale / floor-denied / full playback / interrupted discard paths.
- Intermediate values are exposed: `history_len`, `since_ms`, `record`, `violations`, `kept`, `remaining_queue`, `durations_raw` all appear in the return value; without them, mutants such as parent rewiring or `since` rescaling are invisible in the final result.
- Hand anchors must be unique: each anchor is checked with `SRC.count(old)` before replacement to avoid silently patching the wrong site (the `can-user-floor` case below is exactly this trap).

## Mutation design (40 mutants, all CAUGHT)

- Transition family: no-exit condition / reject reason / dropping 4 key `VALID` pairs / emptying `FLOOR_BLOCKS` or switching it to AGENT / not updating state on acceptance / no history increment;
- Floor family: missing `STOPPED/ERROR` check, allowing `USER` floor, inverted turn_id check, `request_agent_floor` passing an extra turn_id, flipped floor value, `release/handle` and `should_commit` branches;
- Interrupt family: `<`→`<=`, `since*1000`→`*100`, not updating `last_trigger_ts`, wrong floor in the chained reaction, debounced-branch fields;
- Invariants family: shutdown reason, reversed stale text, `USER` floor branch, INV1 text / condition, INV2 condition;
- Latency family: `round(4)`→`round(3)`, `int(*1000)`→`round(*1000)`, inverted marks whitelist;
- Playback family: `chunk_samples` scaling, inverted stale guard, inverted floor-denied guard, `_drain` discarding the wrong turn, off-by-one `qsize`, `played_subchunks` step, shifted interrupt position, flipped `complete`.
- **Non-unique anchor trap**: `if floor == "USER":` occurs twice in the source (the `if` in `_can_agent_speak`, and as a substring of the `elif` in `invariants`), so `SRC.count` returned 2 → the script printed `ANCHOR-ERROR can-user-floor expected 1 found 2` and wrongly counted it as SURVIVED (39/40). Widening the anchor to a three-line contextual fragment and re-running gave **40/40 CAUGHT, 0 SURVIVED**. This was a script bug, not a semantic survivor.
- Truly equivalent mutants removed: none (all 40 are observable in this round).

## Duplicate-avoidance boundaries

- Distinct from `omni-moshi-multistream-delay` (acoustic delay / overwrite chain), `omni-thinker-talker-replay` (talker replay), `inf-ttft-tpot-queue` (queue latency ledger), and `asr-moonshine-stream-verify` (streaming ASR verification): this problem covers the **full-duplex barge-in control plane + turn_id stale suppression + synthesized-chunk discard ledger**.
- The agentic project steps (ReAct / memory ledger / retry breaker) are a different narrative layer, not a state-machine control plane; their wording is not reused here.
- No real audio / VAD / LLM / TTS / thread scheduling is judged; `chunk_samples` is only a deterministic integer ledger.
