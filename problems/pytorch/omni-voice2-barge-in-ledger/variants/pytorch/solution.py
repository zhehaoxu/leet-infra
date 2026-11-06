"""全双工语音 barge-in 状态机与已合成 chunk 丢弃账（order 833）。

回源：``AIIT-GLITCH/voice2``（master 分支，纯 CPU 控制面，无音频 IO）：
  voice2/enums.py                 8 状态 / 3 floor / 14 reason / 4 interrupt source
  voice2/state_controller.py      _VALID 转移表、STOPPED no-exit、floor block、history
  voice2/floor_manager.py         can_agent_speak / request_agent_floor / ...
  voice2/interrupt_controller.py  debounce（首个触发胜出）+ 三连锁反应
  voice2/invariants.py            playback_is_allowed 优先级 + INV1/INV2
  voice2/logging_util.py          LatencyTrace 的 duration / to_dict
  voice2/workers/playback.py     stale guard、chunk 切分、中断时 _drain
  voice2/config.py               chunk_ms=20 / sample_rate=22050 / debounce_ms=200

voice2 是一个「边听边说」的全双工语音引擎：用户随时可以打断（barge-in）助手，
引擎必须把旧 turn 已排队/已合成的 TTS chunk 全部丢弃，避免旧音频漏进新 turn。
本题把这层控制面抽成无状态判题：所有时间、队列、状态、floor、turn_id 都作
显式入参，不跑线程、不碰音频、不调 time.monotonic()。

判题约定：单入口 solve(mode, **kw)；7 个 mode：
  transition  StateController.transition（转移表 / no-exit / floor block / history）
  turn        start_new_turn / snapshot
  floor       FloorManager 五个动作
  interrupt   InterruptController.trigger（debounce + 三连锁）
  invariants  playback_is_allowed 优先级 + check_all 两条不变量
  latency     LatencyTrace 的 duration / to_dict 四个关键耗时
  playback    PlaybackWorker 的 stale guard + 分块 + 中断丢弃账

不进判题域（避免与既有题重复、或需真线程/音频）：线程与锁、sounddevice 输出流、
UDP tap、Silero VAD 能量门限、真实 ASR/LLM/TTS、ring buffer、键盘热键。
"""

STATES = ("IDLE", "LISTENING", "THINKING", "SPEAKING", "INTERRUPTING",
          "FALLBACK_TEXT", "ERROR", "STOPPED")
FLOORS = ("NONE", "USER", "AGENT")
REASONS = ("STARTUP", "SPEECH_DETECTED", "SPEECH_ENDED", "ASR_COMPLETE",
           "THINK_START", "THINK_COMPLETE", "PLAYBACK_START",
           "PLAYBACK_COMPLETE", "INTERRUPT", "INTERRUPT_RESOLVED",
           "ERROR_RECOVERY", "CAPABILITY_FAILURE", "SHUTDOWN", "FALLBACK")
SOURCES = ("KEYBOARD", "VAD", "PROGRAMMATIC", "PARTIAL_ASR")
LATENCY_FIELDS = (
    "turn_start", "speech_detect_start", "speech_detect_end",
    "asr_start", "asr_end", "think_start", "first_token", "think_end",
    "tts_start", "tts_first_sample", "playback_start",
    "interrupt_detected", "interrupt_executed", "playback_end", "turn_end",
)

# 官方 _VALID：只允许下列 (from, to)，共 28 条。
VALID = frozenset({
    ("IDLE", "LISTENING"), ("IDLE", "THINKING"), ("IDLE", "FALLBACK_TEXT"),
    ("IDLE", "ERROR"), ("IDLE", "STOPPED"),
    ("LISTENING", "THINKING"), ("LISTENING", "IDLE"),
    ("LISTENING", "INTERRUPTING"), ("LISTENING", "ERROR"),
    ("LISTENING", "STOPPED"),
    ("THINKING", "SPEAKING"), ("THINKING", "IDLE"),
    ("THINKING", "INTERRUPTING"), ("THINKING", "ERROR"),
    ("THINKING", "STOPPED"),
    ("SPEAKING", "IDLE"), ("SPEAKING", "LISTENING"),
    ("SPEAKING", "INTERRUPTING"), ("SPEAKING", "ERROR"),
    ("SPEAKING", "STOPPED"),
    ("INTERRUPTING", "LISTENING"), ("INTERRUPTING", "IDLE"),
    ("INTERRUPTING", "ERROR"), ("INTERRUPTING", "STOPPED"),
    ("ERROR", "IDLE"), ("ERROR", "STOPPED"),
    ("FALLBACK_TEXT", "THINKING"), ("FALLBACK_TEXT", "STOPPED"),
})

# 官方 _FLOOR_BLOCKS：进入 SPEAKING 时若 floor 是 USER 则拒绝。
FLOOR_BLOCKS = {"SPEAKING": frozenset({"USER"})}

DEBOUNCE_MS = 200
CHUNK_MS = 20
SAMPLE_RATE = 22050
PLAYBACK_QUEUE_MAXSIZE = 200


def _check_names(state, floor, reason=None, source=None):
    if state not in STATES:
        raise ValueError("unknown state")
    if floor not in FLOORS:
        raise ValueError("unknown floor")
    if reason is not None and reason not in REASONS:
        raise ValueError("unknown reason")
    if source is not None and source not in SOURCES:
        raise ValueError("unknown source")


def _transition(state, new_state, floor, reason, turn_id, history_len):
    """逐行复刻 StateController.transition，返回 (accepted, state, why, record)。"""
    _check_names(state, floor, reason=reason)
    if new_state not in STATES:
        raise ValueError("unknown new_state")
    if state == "STOPPED" and new_state != "STOPPED":
        return False, state, "STOPPED_no_exit", None
    if (state, new_state) not in VALID:
        return False, state, "invalid_transition", None
    if floor in FLOOR_BLOCKS.get(new_state, frozenset()):
        return False, state, "floor_blocked", None
    rec = {"from_state": state, "to_state": new_state, "reason": reason,
           "floor_owner": floor, "turn_id": turn_id}
    return True, new_state, None, rec


def _can_agent_speak(state, floor, turn_id, current_turn_id):
    """FloorManager.can_agent_speak：turn_id=0 表示不校验（官方默认 0）。"""
    if state in ("STOPPED", "ERROR"):
        return False
    if floor == "USER":
        return False
    if turn_id and turn_id != current_turn_id:
        return False
    return True


def _duration(marks, a, b):
    if a in marks and b in marks:
        return round(marks[b] - marks[a], 4)
    return None


def solve(mode: str, **kw):
    if mode == "transition":
        state = kw["state"]
        new_state = kw["new_state"]
        floor = kw["floor"]
        reason = kw["reason"]
        turn_id = kw.get("turn_id", 0)
        history_len = kw.get("history_len", 0)
        ok, new_s, why, rec = _transition(state, new_state, floor, reason,
                                          turn_id, history_len)
        return {"accepted": ok, "state": new_s, "reject_reason": why,
                "record": rec, "floor": floor,
                "history_len": history_len + 1 if ok else history_len}

    if mode == "turn":
        action = kw["action"]
        current = kw["current_turn_id"]
        if action == "start_new_turn":
            return {"current_turn_id": current + 1, "turn_id": current + 1}
        if action == "snapshot":
            return {"state": kw["state"], "floor_owner": kw["floor"],
                    "turn_id": current, "history_len": kw.get("history_len", 0)}
        raise ValueError("unknown turn action")

    if mode == "floor":
        action = kw["action"]
        state, floor = kw["state"], kw["floor"]
        turn_id = kw.get("turn_id", 0)
        current = kw.get("current_turn_id", 0)
        speaking = kw.get("speaking", True)
        _check_names(state, floor)
        if action == "can_agent_speak":
            return {"action": action,
                    "allowed": _can_agent_speak(state, floor, turn_id, current),
                    "floor": floor, "speaking": speaking}
        if action == "request_agent_floor":
            # 官方 request_agent_floor 调用 can_agent_speak() 时不传 turn_id
            allowed = _can_agent_speak(state, floor, 0, current)
            return {"action": action, "allowed": allowed,
                    "floor": "AGENT" if allowed else floor,
                    "speaking": speaking}
        if action == "release_floor":
            return {"action": action, "allowed": True,
                    "floor": "NONE", "speaking": speaking}
        if action == "should_commit_user_audio":
            allowed = state not in ("STOPPED", "ERROR") and floor in ("USER", "NONE")
            return {"action": action, "allowed": allowed,
                    "floor": floor, "speaking": speaking}
        if action == "handle_interrupt":
            return {"action": action, "allowed": True,
                    "floor": "USER", "speaking": False}
        raise ValueError("unknown floor action")

    if mode == "interrupt":
        now = float(kw["now"])
        last = float(kw["last_trigger_ts"])
        debounce_ms = kw.get("debounce_ms", DEBOUNCE_MS)
        source = kw.get("source", "VAD")
        state, floor = kw["state"], kw["floor"]
        current = kw.get("current_turn_id", 0)
        history_len = kw.get("history_len", 0)
        _check_names(state, floor, source=source)
        since = now - last
        if since < debounce_ms / 1000.0:
            return {"triggered": False, "since_ms": int(since * 1000),
                    "last_trigger_ts": last, "interrupted": False,
                    "floor": floor, "state": state,
                    "transition_accepted": None, "reject_reason": None,
                    "history_len": history_len, "reason": "debounced"}
        # 三连锁：mark interrupt_detected → set interrupted → floor→USER → state→INTERRUPTING
        ok, new_s, why, rec = _transition(state, "INTERRUPTING", "USER",
                                          "INTERRUPT", current, history_len)
        return {"triggered": True, "since_ms": int(since * 1000),
                "last_trigger_ts": now, "interrupted": True,
                "floor": "USER", "state": new_s,
                "transition_accepted": ok, "reject_reason": why,
                "record": rec,
                "history_len": history_len + 1 if ok else history_len,
                "reason": "triggered", "source": source}

    if mode == "invariants":
        turn_id = kw["turn_id"]
        current = kw["current_turn_id"]
        shutdown = bool(kw.get("shutdown", False))
        interrupted = bool(kw.get("interrupted", False))
        floor = kw["floor"]
        state = kw["state"]
        _check_names(state, floor)
        # InvariantChecker.playback_is_allowed：优先级 shutdown > interrupted >
        # stale_turn > user_owns_floor > ok
        if shutdown:
            allowed, why = False, "shutdown"
        elif interrupted:
            allowed, why = False, "interrupted"
        elif turn_id != current:
            allowed, why = False, "stale_turn:%d!=%d" % (turn_id, current)
        elif floor == "USER":
            allowed, why = False, "user_owns_floor"
        else:
            allowed, why = True, "ok"
        # check_all：INV1 强制修复 floor；INV2 只审计
        violations = []
        repaired = False
        if state == "SPEAKING" and floor != "AGENT":
            violations.append("INV1: SPEAKING but floor=%s, expected AGENT" % floor)
            floor = "AGENT"
            repaired = True
        if interrupted and state == "SPEAKING":
            violations.append("INV2: speaking while interrupted flag is set")
        return {"allowed": allowed, "reason": why,
                "violations": violations, "floor": floor,
                "repaired": repaired}

    if mode == "latency":
        turn_id = kw["turn_id"]
        marks = dict(kw["marks"])
        for k in marks:
            if k not in LATENCY_FIELDS:
                raise ValueError("unknown latency mark: %s" % k)
        d_asr = _duration(marks, "asr_start", "asr_end")
        d_think = _duration(marks, "think_start", "think_end")
        d_stop = _duration(marks, "interrupt_detected", "interrupt_executed")
        d_total = _duration(marks, "turn_start", "turn_end")
        return {
            "turn_id": turn_id,
            "marks": {k: round(v, 4) for k, v in marks.items()},
            "durations_raw": {"asr": d_asr, "think": d_think,
                              "interrupt_stop": d_stop, "total": d_total},
            "durations": {
                "asr_ms": int((d_asr or 0) * 1000),
                "think_ms": int((d_think or 0) * 1000),
                "interrupt_stop_ms": int((d_stop or 0) * 1000),
                "total_turn_ms": int((d_total or 0) * 1000),
            },
        }

    if mode == "playback":
        turn_id = kw["turn_id"]
        current = kw["current_turn_id"]
        sample_rate = kw.get("sample_rate", SAMPLE_RATE)
        chunk_ms = kw.get("chunk_ms", CHUNK_MS)
        queue = [list(x) for x in kw.get("queue", [])]
        subchunks = int(kw["subchunks"])
        interrupt_at = kw.get("interrupt_at", None)
        floor = kw["floor"]
        state = kw["state"]
        _check_names(state, floor)
        chunk_samples = int(sample_rate * chunk_ms / 1000)
        base = {"chunk_samples": chunk_samples, "stale_suppressed": False,
                "floor_denied": False, "transition_accepted": None,
                "state": state, "floor": floor, "played_subchunks": 0,
                "interrupted": False, "queued_chunks_discarded": 0,
                "drained": 0, "kept": [], "remaining_queue": queue,
                "complete": False}
        # 1. stale generation guard：turn 已推进，整段回复作废
        if turn_id != current:
            base["stale_suppressed"] = True
            return base
        # 2. request_agent_floor（USER 持有时拒绝）
        if not _can_agent_speak(state, floor, 0, current):
            base["floor_denied"] = True
            return base
        floor = "AGENT"
        base["floor"] = floor
        # 3. transition → SPEAKING（floor 已是 AGENT，不会被 block）
        ok, new_s, why, rec = _transition(state, "SPEAKING", floor,
                                          "PLAYBACK_START", turn_id,
                                          kw.get("history_len", 0))
        base["transition_accepted"] = ok
        base["state"] = new_s
        base["reject_reason"] = why
        base["record"] = rec
        if not ok:
            return base
        # 4. 逐 subchunk 播放；中断发生在 interrupt_at 指定的 subchunk 起点
        for k in range(subchunks):
            if interrupt_at is not None and k == interrupt_at:
                base["interrupted"] = True
                base["queued_chunks_discarded"] = len(queue)
                # _drain：只丢本 turn 的排队项，保留其它 turn
                kept, drained = [], 0
                for item in queue:
                    if len(item) >= 2 and item[1] == turn_id:
                        drained += 1
                    else:
                        kept.append(item)
                base["drained"] = drained
                base["kept"] = kept
                base["remaining_queue"] = kept
                return base
            base["played_subchunks"] += 1
        base["complete"] = True
        return base

    raise ValueError("unknown mode: %s" % mode)
