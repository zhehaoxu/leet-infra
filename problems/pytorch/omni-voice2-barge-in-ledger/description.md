# 全双工 barge-in 控制面与已合成 chunk 丢弃账

## 背景

`AIIT-GLITCH/voice2` 是一个「边听边说」的全双工语音引擎：用户随时可以打断（barge-in）助手正在播放的回复，引擎必须立刻停掉当前 turn 已排队、已合成的 TTS chunk，否则旧音频会漏进新 turn。控制面由状态机、发言权（floor）、打断去抖、播放许可不变量与延迟账组成。

本题把这层控制面抽成**无状态判题**：所有时间、队列、状态、floor、turn_id 都作显式入参；不跑线程、不碰真实音频、不调用 `time.monotonic()`、不调 LLM / ASR / TTS。实现单入口 `solve(mode, **kw)`，共 7 个 mode。

## 通用约定

枚举取值固定如下，名称非法时抛 `ValueError`：

```
STATES  = IDLE LISTENING THINKING SPEAKING INTERRUPTING FALLBACK_TEXT ERROR STOPPED
FLOORS  = NONE USER AGENT
REASONS = STARTUP SPEECH_DETECTED SPEECH_ENDED ASR_COMPLETE THINK_START
          THINK_COMPLETE PLAYBACK_START PLAYBACK_COMPLETE INTERRUPT
          INTERRUPT_RESOLVED ERROR_RECOVERY CAPABILITY_FAILURE SHUTDOWN FALLBACK
SOURCES = KEYBOARD VAD PROGRAMMATIC PARTIAL_ASR
```

默认常量：`debounce_ms = 200`、`chunk_ms = 20`、`sample_rate = 22050`、`playback_queue_maxsize = 200`。时间入参单位为秒（float）；`since_ms` 与 `*_ms` 一律 `int(x * 1000)` 截断，不用 `round`。

## 模式与规则

所有 mode 返回 dict；未在返回值里列出的中间量不判。

### 1. transition：状态转移

输入 `state, new_state, floor, reason, turn_id=0, history_len=0`。按优先级判定：

1. `state == "STOPPED"` 且 `new_state != "STOPPED"` → 拒绝 `STOPPED_no_exit`；
2. `(state, new_state)` 不在 `VALID` → 拒绝 `invalid_transition`；
3. `floor` 属于 `FLOOR_BLOCKS[new_state]` → 拒绝 `floor_blocked`；
4. 否则接受：`state = new_state`，生成 `record`（`from_state / to_state / reason / floor_owner / turn_id`），`history_len += 1`。

转移表共 28 条，注意 `STOPPED -> STOPPED` 本身不在 `VALID` 中、也不满足第 1 条，故官方返回 `invalid_transition`：

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

返回 `{"accepted", "state", "reject_reason", "record", "floor", "history_len"}`：`floor` 恒为入参原值，`reject_reason` 接受时为 `None`，`record` 拒绝时为 `None`。

### 2. turn：轮次推进与快照

输入 `action, current_turn_id, state, floor, history_len=0`。

- `start_new_turn` → `{"current_turn_id": current + 1, "turn_id": current + 1}`；
- `snapshot` → `{"state", "floor_owner": floor, "turn_id": current_turn_id, "history_len"}`。

其余 action 抛 `ValueError`。

### 3. floor：发言权管理

输入 `action, state, floor, turn_id=0, current_turn_id=0, speaking=True`，返回 `{"action", "allowed", "floor", "speaking"}`。

- `can_agent_speak`：`state` 非 `STOPPED/ERROR`、`floor != "USER"`、且 `turn_id` 为 0 或等于 `current_turn_id` 时允许；`turn_id = 0` 是官方默认的「不校验」。
- `request_agent_floor`：官方调用 `can_agent_speak()` **不传 turn_id**（按 0 处理）；允许则 `floor = "AGENT"`，否则 `floor` 不变。
- `release_floor`：`allowed=True`，`floor = "NONE"`。
- `should_commit_user_audio`：`state` 非 `STOPPED/ERROR` 且 `floor` 属于 `("USER", "NONE")` 时允许，`floor` 不变。
- `handle_interrupt`：`allowed=True`，`floor = "USER"`，`speaking=False`。

其余 action 抛 `ValueError`。

### 4. interrupt：打断去抖与三连锁

输入 `now, last_trigger_ts, debounce_ms=200, source="VAD", state, floor, current_turn_id=0, history_len=0`。令 `since = now - last_trigger_ts`。

`since < debounce_ms / 1000.0` 为去抖：`triggered=False`、`since_ms=int(since*1000)`、`last_trigger_ts` 保持 `last`、`interrupted=False`、`floor/state` 保持原值、`transition_accepted=None`、`reject_reason=None`、`history_len` 不变、`reason="debounced"`；去抖分支不设 `record` 与 `source` 字段。

否则**首个触发胜出**，依次执行三连锁：置 `interrupted` → `floor = "USER"` → 调用 `transition(state, "INTERRUPTING", "USER", "INTERRUPT", current_turn_id, history_len)`。返回 `triggered=True`、`since_ms`、`last_trigger_ts=now`、`interrupted=True`、`floor="USER"`、`state` 为转移后的状态、`transition_accepted`、`reject_reason`、`record`、`history_len`（转移接受才 +1）、`reason="triggered"`、`source`。

**关键**：即使转移被 `VALID` 拒绝（例如从 `IDLE` 打断），三连锁仍然生效——`interrupted` 与 `floor` 已改，只是 `state` 不变、`history_len` 不增。

### 5. invariants：播放许可与不变量

输入 `turn_id, current_turn_id, shutdown=False, interrupted=False, floor, state`。`playback_is_allowed` 按优先级取第一个命中：

```
shutdown                   -> allowed=False, reason="shutdown"
interrupted                -> allowed=False, reason="interrupted"
turn_id != current         -> allowed=False, reason="stale_turn:%d!=%d"
floor == "USER"            -> allowed=False, reason="user_owns_floor"
otherwise                  -> allowed=True,  reason="ok"
```

`check_all` 两条不变量：

- **INV1**：`state == "SPEAKING"` 且 `floor != "AGENT"` → 追加 `"INV1: SPEAKING but floor=%s, expected AGENT" % floor`，并把 `floor` 强制修为 `"AGENT"`、`repaired=True`（强制修复）；
- **INV2**：`interrupted` 且 `state == "SPEAKING"` → 只追加 `"INV2: speaking while interrupted flag is set"`，不修状态（仅审计）。

返回 `{"allowed", "reason", "violations", "floor", "repaired"}`。

### 6. latency：延迟账

输入 `turn_id, marks`（dict）。`marks` 的键只能取自以下 15 个字段，否则抛 `ValueError`：

```
turn_start speech_detect_start speech_detect_end asr_start asr_end
think_start first_token think_end tts_start tts_first_sample
playback_start interrupt_detected interrupt_executed playback_end turn_end
```

输出的 `marks` 为每个值 `round(v, 4)`；`durations_raw` 是 4 段 `round(marks[b] - marks[a], 4)`（缺任一 mark 则为 `None`）：

```
asr            = asr_start            -> asr_end
think          = think_start          -> think_end
interrupt_stop = interrupt_detected   -> interrupt_executed
total          = turn_start           -> turn_end
```

`durations` 把这 4 段乘 1000 后 `int()` 截断，键为 `asr_ms / think_ms / interrupt_stop_ms / total_turn_ms`，缺失段按 0 计。返回 `{"turn_id", "marks", "durations_raw", "durations"}`。

### 7. playback：播放、stale guard 与丢弃账

输入 `turn_id, current_turn_id, subchunks, queue=[], interrupt_at=None, sample_rate=22050, chunk_ms=20, floor, state, history_len=0`。`queue` 元素形如 `[item, turn_id, ...]`；`chunk_samples = int(sample_rate * chunk_ms / 1000)`（默认 `22050 * 20 / 1000 = 441`）。

流程：

1. **stale generation guard**：`turn_id != current_turn_id` → `stale_suppressed=True`，其余字段取默认值，立即返回（整段回复作废，不播、不转移）。
2. **request_agent_floor**：若 `can_agent_speak` 不允许（如 `USER` 持 floor）→ `floor_denied=True`，立即返回。
3. `floor = "AGENT"`；调用 `transition(state, "SPEAKING", floor, "PLAYBACK_START", turn_id, history_len)`；不接受则带 `transition_accepted=False` / `reject_reason` 返回。
4. 逐 subchunk 播放 `for k in range(subchunks)`：若 `interrupt_at is not None and k == interrupt_at` → `interrupted=True`、`queued_chunks_discarded = len(queue)`（`qsize()` 口径，含所有 turn 的排队项），随后 `_drain` **只丢本 turn 的排队项**：

```
kept, drained = [], 0
for item in queue:
    if len(item) >= 2 and item[1] == turn_id:
        drained += 1
    else:
        kept.append(item)
```

   返回 `drained / kept / remaining_queue=kept`（其它 turn 的排队项保留）。否则 `played_subchunks += 1`。
5. 循环走完 `complete=True`。

返回字段：`chunk_samples, stale_suppressed, floor_denied, transition_accepted, state, floor, played_subchunks, interrupted, queued_chunks_discarded, drained, kept, remaining_queue, complete`；转移执行后另有 `reject_reason, record`。`kept` 初值 `[]`、`drained` 初值 `0`、`remaining_queue` 初值等于入参 `queue`（未完成分支保持原样）。

## 输入与返回示例

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

## 不进判题域

线程与锁、`sounddevice` 输出流、UDP tap、VAD 能量门限、真实 ASR / LLM / TTS、ring buffer、键盘热键，以及任何真实音频 IO。判题只考上述控制面的确定性账。
