# Full-duplex barge-in control plane and synthesized-chunk discard ledger

## Background

`AIIT-GLITCH/voice2` is a listen-while-speaking full-duplex voice engine: the user may barge in at any moment while the assistant is playing a reply, and the engine must immediately stop the TTS chunks already queued or synthesized for the current turn, otherwise stale audio leaks into the new turn. The control plane consists of a state machine, a floor owner, interrupt debouncing, playback-permission invariants, and a latency trace.

This problem distills that control plane into **stateless judging**: all time, queues, states, floor owners, and turn ids are explicit inputs; no threads, no real audio, no `time.monotonic()`, no LLM / ASR / TTS calls. Implement a single entry point `solve(mode, **kw)` with 7 modes.

## General conventions

The enums are fixed as follows; an unknown name raises `ValueError`:

```
STATES  = IDLE LISTENING THINKING SPEAKING INTERRUPTING FALLBACK_TEXT ERROR STOPPED
FLOORS  = NONE USER AGENT
REASONS = STARTUP SPEECH_DETECTED SPEECH_ENDED ASR_COMPLETE THINK_START
          THINK_COMPLETE PLAYBACK_START PLAYBACK_COMPLETE INTERRUPT
          INTERRUPT_RESOLVED ERROR_RECOVERY CAPABILITY_FAILURE SHUTDOWN FALLBACK
SOURCES = KEYBOARD VAD PROGRAMMATIC PARTIAL_ASR
```

Default constants: `debounce_ms = 200`, `chunk_ms = 20`, `sample_rate = 22050`, `playback_queue_maxsize = 200`. Time inputs are seconds (float); `since_ms` and `*_ms` are always truncated with `int(x * 1000)`, never `round`.

## Modes and rules

Every mode returns a dict; intermediate values not listed in the return value are not judged.

### 1. transition: state transitions

Inputs: `state, new_state, floor, reason, turn_id=0, history_len=0`. Decisions in priority order:

1. `state == "STOPPED"` and `new_state != "STOPPED"` → reject with `STOPPED_no_exit`;
2. `(state, new_state)` not in `VALID` → reject with `invalid_transition`;
3. `floor` in `FLOOR_BLOCKS[new_state]` → reject with `floor_blocked`;
4. otherwise accept: `state = new_state`, build `record` (`from_state / to_state / reason / floor_owner / turn_id`), and `history_len += 1`.

The transition table has 28 entries. Note that `STOPPED -> STOPPED` is itself absent from `VALID` and does not satisfy rule 1, so the official code returns `invalid_transition`:

```
VALID = {
    IDLE>LISTENING, IDLE>THINKING, IDLE>FALLBACK_TEXT, IDLE>ERROR, IDLE>STOPPED,
    LISTENING>THINKING, LISTENING>IDLE, LISTENING>INTERRUPTING,
    LISTENING>ERROR, LISTENING>STOPPED,
    THINKING>SPEAKING, THINKING>IDLE, THINKING>INTERRUPTING,
    THINKING>ERROR, THINKING>STOPPED,
    SPEAKING>IDLE, SPEAKING>LISTENING, SPEAKING>INTERRUPTING,
    SPEAKING>ERROR, SPEAKING>STOPPED,
    INTERRUPTING>LISTENING, INTERRUPTING>IDLE, INTERRUPTING>ERROR,
    INTERRUPTING>STOPPED,
    ERROR>IDLE, ERROR>STOPPED,
    FALLBACK_TEXT>THINKING, FALLBACK_TEXT>STOPPED,
}
FLOOR_BLOCKS = {SPEAKING: {USER}}
```

Returns `{"accepted", "state", "reject_reason", "record", "floor", "history_len"}`: `floor` is always the input value, `reject_reason` is `None` on acceptance, and `record` is `None` on rejection.

### 2. turn: advancing and snapshotting a turn

Inputs: `action, current_turn_id, state, floor, history_len=0`.

- `start_new_turn` → `{"current_turn_id": current + 1, "turn_id": current + 1}`;
- `snapshot` → `{"state", "floor_owner": floor, "turn_id": current_turn_id, "history_len"}`.

Any other action raises `ValueError`.

### 3. floor: floor-owner management

Inputs: `action, state, floor, turn_id=0, current_turn_id=0, speaking=True`; returns `{"action", "allowed", "floor", "speaking"}`.

- `can_agent_speak`: allowed when `state` is not `STOPPED/ERROR`, `floor != "USER"`, and `turn_id` is 0 or equals `current_turn_id`; `turn_id = 0` is the official default meaning "no check".
- `request_agent_floor`: the official code calls `can_agent_speak()` **without a turn_id** (i.e. as 0); when allowed `floor = "AGENT"`, otherwise `floor` is unchanged.
- `release_floor`: `allowed=True`, `floor = "NONE"`.
- `should_commit_user_audio`: allowed when `state` is not `STOPPED/ERROR` and `floor` is in `("USER", "NONE")`; `floor` is unchanged.
- `handle_interrupt`: `allowed=True`, `floor = "USER"`, `speaking=False`.

Any other action raises `ValueError`.

### 4. interrupt: debouncing and the three chained reactions

Inputs: `now, last_trigger_ts, debounce_ms=200, source="VAD", state, floor, current_turn_id=0, history_len=0`. Let `since = now - last_trigger_ts`.

If `since < debounce_ms / 1000.0`, the trigger is debounced: `triggered=False`, `since_ms=int(since*1000)`, `last_trigger_ts` stays `last`, `interrupted=False`, `floor/state` unchanged, `transition_accepted=None`, `reject_reason=None`, `history_len` unchanged, `reason="debounced"`. The `record` and `source` fields are not set in the debounced branch.

Otherwise **the first trigger wins** and three reactions run in order: set `interrupted` → `floor = "USER"` → call `transition(state, "INTERRUPTING", "USER", "INTERRUPT", current_turn_id, history_len)`. Returns `triggered=True`, `since_ms`, `last_trigger_ts=now`, `interrupted=True`, `floor="USER"`, `state` after the transition, `transition_accepted`, `reject_reason`, `record`, `history_len` (incremented only when the transition is accepted), `reason="triggered"`, `source`.

**Key point**: even when the transition is rejected by `VALID` (for example interrupting from `IDLE`), the three reactions still take effect — `interrupted` and `floor` have already changed; only `state` stays put and `history_len` does not grow.

### 5. invariants: playback permission and invariants

Inputs: `turn_id, current_turn_id, shutdown=False, interrupted=False, floor, state`. `playback_is_allowed` takes the first matching branch:

```
shutdown                   -> allowed=False, reason="shutdown"
interrupted                -> allowed=False, reason="interrupted"
turn_id != current         -> allowed=False, reason="stale_turn:%d!=%d"
floor == "USER"            -> allowed=False, reason="user_owns_floor"
otherwise                  -> allowed=True,  reason="ok"
```

Two invariants in `check_all`:

- **INV1**: `state == "SPEAKING"` and `floor != "AGENT"` → append `"INV1: SPEAKING but floor=%s, expected AGENT" % floor`, force `floor = "AGENT"`, and set `repaired=True` (a forced repair);
- **INV2**: `interrupted` and `state == "SPEAKING"` → only append `"INV2: speaking while interrupted flag is set"`, without repairing the state (audit only).

Returns `{"allowed", "reason", "violations", "floor", "repaired"}`.

### 6. latency: latency ledger

Inputs: `turn_id, marks` (dict). Keys of `marks` may only come from the following 15 fields, otherwise `ValueError` is raised:

```
turn_start speech_detect_start speech_detect_end asr_start asr_end
think_start first_token think_end tts_start tts_first_sample
playback_start interrupt_detected interrupt_executed playback_end turn_end
```

The output `marks` applies `round(v, 4)` to each value; `durations_raw` holds 4 spans computed as `round(marks[b] - marks[a], 4)` (or `None` when either mark is missing):

```
asr            = asr_start            -> asr_end
think          = think_start          -> think_end
interrupt_stop = interrupt_detected   -> interrupt_executed
total          = turn_start           -> turn_end
```

`durations` multiplies the 4 spans by 1000 and truncates with `int()`, under keys `asr_ms / think_ms / interrupt_stop_ms / total_turn_ms`, with missing spans counted as 0. Returns `{"turn_id", "marks", "durations_raw", "durations"}`.

### 7. playback: playback, stale guard, and the discard ledger

Inputs: `turn_id, current_turn_id, subchunks, queue=[], interrupt_at=None, sample_rate=22050, chunk_ms=20, floor, state, history_len=0`. Each `queue` item looks like `[item, turn_id, ...]`; `chunk_samples = int(sample_rate * chunk_ms / 1000)` (default `22050 * 20 / 1000 = 441`).

Flow:

1. **stale generation guard**: `turn_id != current_turn_id` → `stale_suppressed=True`, all other fields keep their defaults, return immediately (the whole reply is void: no playback, no transition).
2. **request_agent_floor**: if `can_agent_speak` is not allowed (e.g. `USER` owns the floor) → `floor_denied=True`, return immediately.
3. `floor = "AGENT"`; call `transition(state, "SPEAKING", floor, "PLAYBACK_START", turn_id, history_len)`; if rejected, return with `transition_accepted=False` / `reject_reason`.
4. Play subchunk by subchunk with `for k in range(subchunks)`: if `interrupt_at is not None and k == interrupt_at` → `interrupted=True`, `queued_chunks_discarded = len(queue)` (the `qsize()` figure, counting items of all turns), then `_drain` **discards only the queued items of this turn**:

```
kept, drained = [], 0
for item in queue:
    if len(item) >= 2 and item[1] == turn_id:
        drained += 1
    else:
        kept.append(item)
```

   Returns `drained / kept / remaining_queue=kept` (items of other turns are preserved). Otherwise `played_subchunks += 1`.
5. If the loop finishes, `complete=True`.

Returned fields: `chunk_samples, stale_suppressed, floor_denied, transition_accepted, state, floor, played_subchunks, interrupted, queued_chunks_discarded, drained, kept, remaining_queue, complete`; after the transition is attempted there are also `reject_reason, record`. Defaults: `kept = []`, `drained = 0`, `remaining_queue` equals the input `queue` (unchanged on early-return branches).

## Input and output examples

```
solve(mode="transition", state="THINKING", new_state="SPEAKING",
      floor="AGENT", reason="PLAYBACK_START", turn_id=7, history_len=3)
# {'accepted': True, 'state': 'SPEAKING', 'reject_reason': None,
#  'record': {'from_state': 'THINKING', 'to_state': 'SPEAKING',
#             'reason': 'PLAYBACK_START', 'floor_owner': 'AGENT', 'turn_id': 7},
#  'floor': 'AGENT', 'history_len': 4}

solve(mode="playback", turn_id=7, current_turn_id=7, subchunks=5,
      interrupt_at=2, floor="NONE", state="THINKING",
      queue=[["a", 7], ["b", 6], ["c", 7]])
# played_subchunks=2, interrupted=True, queued_chunks_discarded=3,
# drained=2, kept=[['b', 6]], remaining_queue=[['b', 6]], complete=False
```

## Out of scope

Threads and locks, `sounddevice` output streams, UDP taps, VAD energy thresholds, real ASR / LLM / TTS, ring buffers, keyboard hotkeys, and any real audio IO. Judging covers only the deterministic ledger of the control plane above.
