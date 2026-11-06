"""全双工 barge-in 控制面与已合成 chunk 丢弃账（starter）。

实现 solve(mode, **kw)，通过全部用例。模式清单：
  transition  状态转移：28 条 VALID 表 / STOPPED no-exit / floor_blocked / history
  turn        start_new_turn / snapshot
  floor       can_agent_speak / request_agent_floor / release_floor /
              should_commit_user_audio / handle_interrupt
  interrupt   debounce 去抖（首个触发胜出）+ interrupted/floor/state 三连锁
  invariants  playback_is_allowed 五级优先级 + check_all 的 INV1/INV2
  latency     LatencyTrace 的 round(4) 延迟段与 int(*1000) 毫秒账
  playback    stale guard / request_agent_floor / 逐 subchunk 播放 /
              中断时 qsize 记账 + _drain 只丢本 turn 排队项

各 mode 的参数语义见题面「模式与规则」，常量清单见 solution 同名常量。
判题环境只有 torch + numpy；本题为纯 Python 控制面，不跑线程、不碰音频。
"""

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def solve(mode: str, **kw):
    raise NotImplementedError
