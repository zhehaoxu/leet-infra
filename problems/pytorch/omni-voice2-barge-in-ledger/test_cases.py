"""全双工 barge-in 状态机与已合成 chunk 丢弃账——判题用例（order 833）。

独立 oracle 用纯 Python 从规则重写，不 import solution。
手算锚点：
  - 转移表 29 条、STOPPED no-exit、USER 持 floor 时不许进 SPEAKING；
  - debounce_ms=200 → since=0.15 去抖、since=0.25 放行；
  - playback_is_allowed 五级优先级；
  - LatencyTrace 的 int(duration*1000) 截断（0.1234 → 123 ms）；
  - chunk_samples = int(22050*20/1000) = 441；_drain 只丢本 turn 项。
"""

# ---------------------------------------------------------------- 独立 oracle
# 官方 _VALID，用「from>to」字符串独立抄写（与 solution 的 set 表示不同）
_VALID_ORA = """
IDLE>LISTENING IDLE>THINKING IDLE>FALLBACK_TEXT IDLE>ERROR IDLE>STOPPED
LISTENING>THINKING LISTENING>IDLE LISTENING>INTERRUPTING LISTENING>ERROR LISTENING>STOPPED
THINKING>SPEAKING THINKING>IDLE THINKING>INTERRUPTING THINKING>ERROR THINKING>STOPPED
SPEAKING>IDLE SPEAKING>LISTENING SPEAKING>INTERRUPTING SPEAKING>ERROR SPEAKING>STOPPED
INTERRUPTING>LISTENING INTERRUPTING>IDLE INTERRUPTING>ERROR INTERRUPTING>STOPPED
ERROR>IDLE ERROR>STOPPED
FALLBACK_TEXT>THINKING FALLBACK_TEXT>STOPPED
""".split()
VALID_ORA = set(tuple(p.split(">")) for p in _VALID_ORA)
assert len(VALID_ORA) == 28, len(VALID_ORA)

STATES_ORA = ("IDLE", "LISTENING", "THINKING", "SPEAKING", "INTERRUPTING",
              "FALLBACK_TEXT", "ERROR", "STOPPED")
FLOORS_ORA = ("NONE", "USER", "AGENT")


def ref_transition(state, new_state, floor, reason, turn_id, history_len):
    if state == "STOPPED" and new_state != "STOPPED":
        return False, state, "STOPPED_no_exit", None, history_len
    if (state, new_state) not in VALID_ORA:
        return False, state, "invalid_transition", None, history_len
    if new_state == "SPEAKING" and floor == "USER":
        return False, state, "floor_blocked", None, history_len
    rec = {"from_state": state, "to_state": new_state, "reason": reason,
           "floor_owner": floor, "turn_id": turn_id}
    return True, new_state, None, rec, history_len + 1


def ref_can(state, floor, turn_id, current):
    if state in ("STOPPED", "ERROR"):
        return False
    if floor == "USER":
        return False
    if turn_id and turn_id != current:
        return False
    return True


def ref_floor(action, state, floor, turn_id, current, speaking=True):
    if action == "can_agent_speak":
        return ref_can(state, floor, turn_id, current), floor, speaking
    if action == "request_agent_floor":
        ok = ref_can(state, floor, 0, current)   # 官方不传 turn_id
        return ok, ("AGENT" if ok else floor), speaking
    if action == "release_floor":
        return True, "NONE", speaking
    if action == "should_commit_user_audio":
        return (state not in ("STOPPED", "ERROR") and floor in ("USER", "NONE")), floor, speaking
    if action == "handle_interrupt":
        return True, "USER", False
    raise ValueError(action)


def ref_playback_allowed(turn_id, current, shutdown, interrupted, floor):
    if shutdown:
        return False, "shutdown"
    if interrupted:
        return False, "interrupted"
    if turn_id != current:
        return False, "stale_turn:%d!=%d" % (turn_id, current)
    if floor == "USER":
        return False, "user_owns_floor"
    return True, "ok"


def ref_invariants(state, floor, interrupted):
    v, repaired = [], False
    if state == "SPEAKING" and floor != "AGENT":
        v.append("INV1: SPEAKING but floor=%s, expected AGENT" % floor)
        floor, repaired = "AGENT", True
    if interrupted and state == "SPEAKING":
        v.append("INV2: speaking while interrupted flag is set")
    return v, floor, repaired


def ref_latency(marks):
    def dur(a, b):
        if a in marks and b in marks:
            return round(marks[b] - marks[a], 4)
        return None
    pairs = (("asr", "asr_start", "asr_end"),
             ("think", "think_start", "think_end"),
             ("interrupt_stop", "interrupt_detected", "interrupt_executed"),
             ("total", "turn_start", "turn_end"))
    raw = {n: dur(a, b) for n, a, b in pairs}
    dur_ms = {"asr_ms": int((raw["asr"] or 0) * 1000),
              "think_ms": int((raw["think"] or 0) * 1000),
              "interrupt_stop_ms": int((raw["interrupt_stop"] or 0) * 1000),
              "total_turn_ms": int((raw["total"] or 0) * 1000)}
    return {"marks": {k: round(v, 4) for k, v in marks.items()},
            "durations_raw": raw, "durations": dur_ms}


def ref_playback(turn_id, current, queue, subchunks, interrupt_at, floor, state,
                 sample_rate=22050, chunk_ms=20):
    cs = int(sample_rate * chunk_ms / 1000)
    out = {"chunk_samples": cs, "stale_suppressed": False,
           "floor_denied": False, "transition_accepted": None,
           "state": state, "floor": floor, "played_subchunks": 0,
           "interrupted": False, "queued_chunks_discarded": 0,
           "drained": 0, "kept": [], "remaining_queue": list(queue),
           "complete": False}
    if turn_id != current:
        out["stale_suppressed"] = True
        return out
    if not ref_can(state, floor, 0, current):
        out["floor_denied"] = True
        return out
    floor = "AGENT"
    ok, ns, why, _rec, _h = ref_transition(state, "SPEAKING", floor,
                                           "PLAYBACK_START", turn_id, 0)
    out["floor"] = floor
    out["transition_accepted"] = ok
    out["state"] = ns
    out["reject_reason"] = why
    if not ok:
        return out
    for k in range(subchunks):
        if interrupt_at is not None and k == interrupt_at:
            out["interrupted"] = True
            out["queued_chunks_discarded"] = len(queue)
            kept = [it for it in queue if not (len(it) >= 2 and it[1] == turn_id)]
            out["drained"] = len(queue) - len(kept)
            out["kept"] = kept
            out["remaining_queue"] = kept
            return out
        out["played_subchunks"] += 1
    out["complete"] = True
    return out


def eq(got, exp, tag=""):
    assert got == exp, "%s: got %r, expected %r" % (tag, got, exp)


# ---------------------------------------------------------------- checks

def check_transition(solve):
    """转移表 29 条 + STOPPED no-exit + floor block + history 记录。"""
    # 手算锚点：IDLE→THINKING 是 submit_text 旁路（官方注释明写）
    r = solve(mode="transition", state="IDLE", new_state="THINKING",
              floor="NONE", reason="ASR_COMPLETE", turn_id=7, history_len=3)
    eq(r["accepted"], True, "IDLE>THINKING accepted")
    eq(r["state"], "THINKING", "IDLE>THINKING state")
    eq(r["history_len"], 4, "history +1")
    eq(r["record"], {"from_state": "IDLE", "to_state": "THINKING",
                     "reason": "ASR_COMPLETE", "floor_owner": "NONE",
                     "turn_id": 7}, "record")
    # 手算：USER 持有 floor 时不许进 SPEAKING（THINKING>SPEAKING 本在表内）
    r = solve(mode="transition", state="THINKING", new_state="SPEAKING",
              floor="USER", reason="PLAYBACK_START", turn_id=1, history_len=0)
    eq((r["accepted"], r["reject_reason"], r["state"], r["history_len"]),
       (False, "floor_blocked", "THINKING", 0), "floor block")
    # 手算：STOPPED 除 STOPPED 外一律拒绝（no-exit 优先于 invalid_transition）；
    # STOPPED>STOPPED 不在 _VALID 里，故仍以 invalid_transition 拒绝
    r = solve(mode="transition", state="STOPPED", new_state="STOPPED",
              floor="NONE", reason="SHUTDOWN", turn_id=1, history_len=0)
    eq((r["accepted"], r["reject_reason"]), (False, "invalid_transition"),
       "STOPPED>STOPPED not in _VALID")
    r = solve(mode="transition", state="STOPPED", new_state="IDLE",
              floor="NONE", reason="STARTUP", turn_id=1, history_len=5)
    eq((r["accepted"], r["reject_reason"], r["state"], r["history_len"]),
       (False, "STOPPED_no_exit", "STOPPED", 5), "no-exit priority")
    # 全 8×8×3 网格对拍独立 oracle
    n_ok = 0
    for s in STATES_ORA:
        for t in STATES_ORA:
            for f in FLOORS_ORA:
                exp = ref_transition(s, t, f, "STARTUP", 4, 2)
                r = solve(mode="transition", state=s, new_state=t, floor=f,
                          reason="STARTUP", turn_id=4, history_len=2)
                eq((r["accepted"], r["state"], r["reject_reason"],
                    r["history_len"]), (exp[0], exp[1], exp[2], exp[4]),
                   "grid %s>%s/%s" % (s, t, f))
                if r["accepted"]:
                    eq(r["record"], exp[3], "grid record %s>%s" % (s, t))
                    n_ok += 1
                else:
                    eq(r["record"], None, "no record on reject")
    # 28 条 × 3 floor，其中 USER 持 floor 时 THINKING>SPEAKING 被 block
    eq(n_ok, 83, "accepted count (28 pairs x 3 floors minus USER->SPEAKING)")
    # 非法输入
    for bad in ("WALKING", "speaking"):
        try:
            solve(mode="transition", state=bad, new_state="IDLE",
                  floor="NONE", reason="STARTUP")
            raise AssertionError("expected ValueError for state")
        except ValueError:
            pass


def check_turn(solve):
    """start_new_turn 自增 + snapshot 原子读。"""
    eq(solve(mode="turn", action="start_new_turn", current_turn_id=0),
       {"current_turn_id": 1, "turn_id": 1}, "first turn")
    eq(solve(mode="turn", action="start_new_turn", current_turn_id=41),
       {"current_turn_id": 42, "turn_id": 42}, "turn 42")
    eq(solve(mode="turn", action="snapshot", current_turn_id=9,
             state="SPEAKING", floor="AGENT", history_len=6),
       {"state": "SPEAKING", "floor_owner": "AGENT", "turn_id": 9,
        "history_len": 6}, "snapshot")
    try:
        solve(mode="turn", action="rewind", current_turn_id=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def check_floor(solve):
    """FloorManager 五动作；request_agent_floor 不校验 turn_id 是官方口径陷阱。"""
    actions = ("can_agent_speak", "request_agent_floor", "release_floor",
               "should_commit_user_audio", "handle_interrupt")
    for a in actions:
        for s in STATES_ORA:
            for f in FLOORS_ORA:
                for tid, cur in ((0, 3), (3, 3), (2, 3)):
                    exp = ref_floor(a, s, f, tid, cur, speaking=True)
                    r = solve(mode="floor", action=a, state=s, floor=f,
                              turn_id=tid, current_turn_id=cur, speaking=True)
                    eq((r["allowed"], r["floor"], r["speaking"]), exp,
                       "%s %s/%s t=%s c=%s" % (a, s, f, tid, cur))
    # 手算：can_agent_speak 传 turn_id=2 但 current=3 → False（陈旧 turn）
    r = solve(mode="floor", action="can_agent_speak", state="THINKING",
              floor="NONE", turn_id=2, current_turn_id=3)
    eq(r["allowed"], False, "stale turn_id blocks agent")
    # 手算：request_agent_floor 同样场景仍放行（官方不传 turn_id）
    r = solve(mode="floor", action="request_agent_floor", state="THINKING",
              floor="NONE", turn_id=2, current_turn_id=3)
    eq((r["allowed"], r["floor"]), (True, "AGENT"), "request_agent_floor ignores turn_id")
    # 手算：handle_interrupt 清 speaking 且把 floor 交回 USER
    r = solve(mode="floor", action="handle_interrupt", state="SPEAKING",
              floor="AGENT", speaking=True)
    eq((r["allowed"], r["floor"], r["speaking"]), (True, "USER", False),
       "handle_interrupt")
    try:
        solve(mode="floor", action="grab", state="IDLE", floor="NONE")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def check_interrupt(solve):
    """debounce 首个触发胜出 + 三连锁；since 用严格小于。"""
    # 手算：since = 0.25 < 0.4（debounce_ms=400）→ 去抖；0.25 精确可表示，
    # int(0.25*1000)=250，避开 0.15 这类二进制浮点的 149 截断
    r = solve(mode="interrupt", now=0.25, last_trigger_ts=0.0,
              debounce_ms=400, source="KEYBOARD", state="SPEAKING",
              floor="AGENT", current_turn_id=5, history_len=2)
    eq((r["triggered"], r["since_ms"], r["last_trigger_ts"], r["interrupted"],
        r["state"], r["history_len"]),
       (False, 250, 0.0, False, "SPEAKING", 2), "debounced")
    # 手算：since = 3.25-3.00 = 0.25 ≥ 0.2 → 触发；三连锁
    r = solve(mode="interrupt", now=3.25, last_trigger_ts=3.00,
              source="VAD", state="SPEAKING", floor="AGENT",
              current_turn_id=5, history_len=2)
    eq((r["triggered"], r["since_ms"], r["last_trigger_ts"], r["interrupted"],
        r["floor"], r["state"], r["transition_accepted"],
        r["reject_reason"], r["history_len"], r["reason"], r["source"]),
       (True, 250, 3.25, True, "USER", "INTERRUPTING", True, None, 3,
        "triggered", "VAD"), "accepted trigger")
    eq(r["record"], {"from_state": "SPEAKING", "to_state": "INTERRUPTING",
                     "reason": "INTERRUPT", "floor_owner": "USER",
                     "turn_id": 5}, "trigger record")
    # 手算：恰好等于 debounce 边界（since == 0.2）不算去抖
    r = solve(mode="interrupt", now=0.20, last_trigger_ts=0.00,
              source="PROGRAMMATIC", state="LISTENING", floor="NONE",
              current_turn_id=1, history_len=0)
    eq(r["triggered"], True, "since == debounce not debounced")
    # 手算：IDLE→INTERRUPTING 不在 _VALID，三连锁仍执行但转移被拒
    r = solve(mode="interrupt", now=9.0, last_trigger_ts=0.0,
              source="VAD", state="IDLE", floor="NONE",
              current_turn_id=1, history_len=0)
    eq((r["triggered"], r["floor"], r["state"],
        r["transition_accepted"], r["reject_reason"], r["history_len"]),
       (True, "USER", "IDLE", False, "invalid_transition", 0),
       "trigger with rejected transition")
    # 自定义 debounce_ms 与多组 (now,last) 对拍
    for dms in (100, 200, 500):
        for now, last in ((0.0, 0.0), (0.05, 0.0), (0.3, 0.0),
                          (1.0, 0.7), (2.5, 1.0), (10.0, 9.95)):
            since = now - last
            exp_trig = not (since < dms / 1000.0)
            r = solve(mode="interrupt", now=now, last_trigger_ts=last,
                      debounce_ms=dms, source="VAD", state="THINKING",
                      floor="NONE", current_turn_id=1, history_len=0)
            eq(r["triggered"], exp_trig, "debounce grid %s %s %s" % (dms, now, last))
            eq(r["since_ms"], int(since * 1000), "since_ms grid")
    try:
        solve(mode="interrupt", now=1.0, last_trigger_ts=0.0,
              source="TELEPATHY", state="IDLE", floor="NONE")
        raise AssertionError("expected ValueError for source")
    except ValueError:
        pass


def check_invariants(solve):
    """playback_is_allowed 五级优先级 + INV1 强制修复 / INV2 审计。"""
    # 手算优先级：四条同时命中时只报 shutdown
    r = solve(mode="invariants", turn_id=2, current_turn_id=3,
              shutdown=True, interrupted=True, floor="USER", state="SPEAKING")
    eq((r["allowed"], r["reason"]), (False, "shutdown"), "shutdown first")
    eq(r["violations"], ["INV1: SPEAKING but floor=USER, expected AGENT",
                         "INV2: speaking while interrupted flag is set"],
       "both violations")
    eq((r["floor"], r["repaired"]), ("AGENT", True), "INV1 forced repair")
    # 手算：stale_turn 文本含 turn_id 与 current_turn_id
    r = solve(mode="invariants", turn_id=7, current_turn_id=9,
              floor="AGENT", state="THINKING")
    eq((r["allowed"], r["reason"]), (False, "stale_turn:7!=9"), "stale")
    # 手算：user_owns_floor 在 turn 相等时生效
    r = solve(mode="invariants", turn_id=9, current_turn_id=9,
              floor="USER", state="LISTENING")
    eq((r["allowed"], r["reason"]), (False, "user_owns_floor"), "floor")
    r = solve(mode="invariants", turn_id=9, current_turn_id=9,
              floor="AGENT", state="SPEAKING")
    eq((r["allowed"], r["reason"], r["violations"], r["repaired"]),
       (True, "ok", [], False), "ok")
    # 全网格对拍
    for tid, cur in ((1, 1), (1, 2)):
        for sh, it in ((False, False), (False, True), (True, False), (True, True)):
            for f in FLOORS_ORA:
                for s in STATES_ORA:
                    exp = ref_playback_allowed(tid, cur, sh, it, f)
                    v, ef, rep = ref_invariants(s, f, it)
                    r = solve(mode="invariants", turn_id=tid,
                              current_turn_id=cur, shutdown=sh,
                              interrupted=it, floor=f, state=s)
                    eq((r["allowed"], r["reason"]), exp,
                       "grid %s/%s %s/%s %s %s" % (tid, cur, sh, it, f, s))
                    eq((r["violations"], r["floor"], r["repaired"]),
                       (v, ef, rep), "inv grid %s %s %s" % (f, s, it))


def check_latency(solve):
    """LatencyTrace：round(4) 与 int(ms) 截断、缺 mark 时 0。"""
    # 手算：asr 0.1234→123ms、think 0.2000→200ms、stop 0.0099→9ms、total 1.5→1500ms
    marks = {"turn_start": 10.0, "asr_start": 10.1, "asr_end": 10.2234,
             "think_start": 10.3, "think_end": 10.5,
             "interrupt_detected": 11.0, "interrupt_executed": 11.0099,
             "turn_end": 11.5}
    r = solve(mode="latency", turn_id=4, marks=marks)
    eq(r["turn_id"], 4, "turn_id")
    eq(r["durations_raw"]["asr"], 0.1234, "asr raw")
    eq(r["durations_raw"]["think"], 0.2, "think raw")
    eq(r["durations_raw"]["interrupt_stop"], 0.0099, "stop raw")
    eq(r["durations_raw"]["total"], 1.5, "total raw")
    eq(r["durations"], {"asr_ms": 123, "think_ms": 200,
                        "interrupt_stop_ms": 9, "total_turn_ms": 1500},
       "durations ms")
    eq(r["marks"]["asr_start"], 10.1, "marks rounded")
    # 手算：缺 mark → None / 0
    r = solve(mode="latency", turn_id=1, marks={"turn_start": 0.0})
    eq(r["durations_raw"], {"asr": None, "think": None,
                            "interrupt_stop": None, "total": None}, "no marks")
    eq(r["durations"], {"asr_ms": 0, "think_ms": 0,
                        "interrupt_stop_ms": 0, "total_turn_ms": 0}, "zero ms")
    # 多组随机但固定的 marks 对拍独立 oracle
    cases = [
        {},
        {"asr_start": 1.0, "asr_end": 1.0009},
        {"think_start": 2.0, "think_end": 2.3456, "turn_start": 1.0,
         "turn_end": 9.8765},
        {"interrupt_detected": 3.0, "interrupt_executed": 3.0001,
         "turn_start": 0.0, "turn_end": 0.0001},
        {k: float(i) / 7 for i, k in enumerate(
            ("turn_start", "speech_detect_start", "speech_detect_end",
             "asr_start", "asr_end", "think_start", "first_token",
             "think_end", "tts_start", "tts_first_sample", "playback_start",
             "interrupt_detected", "interrupt_executed", "playback_end",
             "turn_end"))},
    ]
    for m in cases:
        r = solve(mode="latency", turn_id=2, marks=m)
        exp = ref_latency(m)
        eq(r["marks"], exp["marks"], "latency marks")
        eq(r["durations_raw"], exp["durations_raw"], "latency raw")
        eq(r["durations"], exp["durations"], "latency ms")
    try:
        solve(mode="latency", turn_id=1, marks={"bogus": 1.0})
        raise AssertionError("expected ValueError for unknown mark")
    except ValueError:
        pass


def check_playback(solve):
    """stale guard、chunk_samples、USER 持 floor 拒绝、中断 _drain 只丢本 turn。"""
    # 手算：stale（turn 已推进）整段作废
    r = solve(mode="playback", turn_id=3, current_turn_id=4,
              state="THINKING", floor="NONE", queue=[["a", 3]],
              subchunks=5, interrupt_at=2)
    eq((r["stale_suppressed"], r["played_subchunks"], r["drained"],
        r["complete"]), (True, 0, 0, False), "stale suppressed")
    # 手算：USER 持 floor → floor_denied，不进 SPEAKING
    r = solve(mode="playback", turn_id=3, current_turn_id=3,
              state="THINKING", floor="USER", queue=[], subchunks=5)
    eq((r["floor_denied"], r["played_subchunks"], r["state"]),
       (True, 0, "THINKING"), "floor denied")
    # 手算：完整播放 6 个 subchunk，chunk_samples = int(22050*20/1000) = 441
    r = solve(mode="playback", turn_id=3, current_turn_id=3,
              state="THINKING", floor="NONE", queue=[], subchunks=6)
    eq((r["chunk_samples"], r["transition_accepted"], r["state"], r["floor"],
        r["played_subchunks"], r["interrupted"], r["complete"]),
       (441, True, "SPEAKING", "AGENT", 6, False, True), "full playback")
    # 手算：中断发生在 subchunk 2，已播 2 个；队列 5 项中 3 项属本 turn → drained=3，
    # 其余 2 项保留；queued_chunks_discarded 记中断瞬间 qsize=5
    q = [["a", 3], ["b", 3], ["c", 4], ["d", 3], ["e", 5]]
    r = solve(mode="playback", turn_id=3, current_turn_id=3,
              state="THINKING", floor="NONE", queue=q,
              subchunks=8, interrupt_at=2)
    eq((r["interrupted"], r["played_subchunks"],
        r["queued_chunks_discarded"], r["drained"], r["complete"]),
       (True, 2, 5, 3, False), "interrupt ledger")
    eq(r["kept"], [["c", 4], ["e", 5]], "kept other turns")
    eq(r["remaining_queue"], [["c", 4], ["e", 5]], "remaining queue")
    # 手算：中断在 subchunk 0 → 一个都没播
    r = solve(mode="playback", turn_id=1, current_turn_id=1,
              state="THINKING", floor="NONE", queue=[["x", 1]],
              subchunks=4, interrupt_at=0)
    eq((r["interrupted"], r["played_subchunks"], r["drained"]),
       (True, 0, 1), "interrupt at 0")
    # 手算：自定义 sample_rate/chunk_ms → int(16000*32/1000)=512
    r = solve(mode="playback", turn_id=1, current_turn_id=1,
              state="THINKING", floor="NONE", queue=[], subchunks=1,
              sample_rate=16000, chunk_ms=32)
    eq(r["chunk_samples"], 512, "custom chunk_samples")
    # 全网格对拍独立 oracle
    for tid, cur in ((1, 1), (1, 2)):
        for f in FLOORS_ORA:
            for s in ("IDLE", "THINKING", "SPEAKING", "STOPPED"):
                for at in (None, 0, 1, 3):
                    q2 = [["a", tid], ["b", cur + 1], ["c", tid]]
                    exp = ref_playback(tid, cur, q2, 5, at, f, s)
                    r = solve(mode="playback", turn_id=tid,
                              current_turn_id=cur, state=s, floor=f,
                              queue=q2, subchunks=5, interrupt_at=at)
                    for k in ("chunk_samples", "stale_suppressed",
                              "floor_denied", "transition_accepted", "state",
                              "floor", "played_subchunks", "interrupted",
                              "queued_chunks_discarded", "drained", "kept",
                              "remaining_queue", "complete"):
                        eq(r[k], exp[k],
                           "pb %s/%s %s %s at=%s key=%s" % (tid, cur, f, s, at, k))


def check(solve):
    check_transition(solve)
    check_turn(solve)
    check_floor(solve)
    check_interrupt(solve)
    check_invariants(solve)
    check_latency(solve)
    check_playback(solve)
    print("all checks passed")
