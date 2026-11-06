# 踩坑与回源记录（order 833）

## 回源口径

- 仓 `AIIT-GLITCH/voice2`，默认分支是 **`master`**（`main` 的 tree API 直接 404）；`raw.githubusercontent.com` 多次超时 / 502，改用 GitHub API `contents` + `base64.b64decode` 下载到 `/tmp/voice2_src/`。
- 只用纯 CPU 控制面源码，stub 掉 `sounddevice.OutputStream`，patch `logging_util.event` 与 `time.monotonic` 为可控时钟；不跑线程、不碰音频。

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

- 官方 `playback.py` 用 `from ..enums import`，必须放进 `voice2/workers/` 子包才能 import；放根目录会报 `attempted relative import beyond top-level package`。
- 官方 `start_new_turn()` 会先自增 `current_turn_id` 再让 `snapshot` 读，因此 snapshot 对拍要用自增后的值作 `current_turn_id` 入参。
- 回源对拍脚本 `/tmp/verify833.py` 用 `.venv-eval/bin/python` 跑（`.venv-backend` 无 numpy）：transition 192 + turn 4 + floor 360 + interrupt 270 + invariants 192 + latency 6 + playback/_speak 108 组，外加 `chunk_samples / _split / is_stale`，**1132 组全一致**。

## 机制要点

- **`_VALID` 是 28 条不是 29 条**：首版手数错一数，oracle 断言 `len == 28` 后暴露；也连带修正 `n_ok == 83`（28 组合 × 3 floor 减去 `USER` 持 floor 进 `SPEAKING` 被 block 的 1 条）。
- **`STOPPED -> STOPPED` 返回 `invalid_transition`**：no-exit 条件只拦 `new_state != STOPPED` 的转移，而 `STOPPED -> STOPPED` 本身不在 `_VALID`；首版手算锚点误写成接受，judge 直接报 5/7。
- **三连锁不可回滚**：`InterruptController.trigger` 先 `interrupted.set()`、`handle_interrupt` 把 floor 置 `USER`，最后才 `transition(INTERRUPTING)`；即使转移被 `VALID` 拒绝（如 `IDLE` 打断），`interrupted` 与 `floor` 已经改了，`state` 与 `history_len` 才保持不变。判题必须把三者分开断言。
- **`request_agent_floor` 调用 `can_agent_speak()` 不传 `turn_id`**（官方默认 0）：这是全题最容易写错的一处，专设变异 `request-agent-turn-id` 抓「多传一个 turn_id」。
- **`playback_is_allowed` 五级优先级**：`shutdown > interrupted > stale_turn > user_owns_floor > ok`；`stale_turn` 文本是 `"stale_turn:%d!=%d" % (turn_id, current)`，顺序不能反。
- **INV1 强制修复、INV2 仅审计**：SPEAKING 且 floor 非 AGENT 时把 floor 改成 AGENT 并置 `repaired=True`；interrupted 且 SPEAKING 时只追加 violations，不改状态。
- **延迟账双重取整**：`LatencyTrace.to_dict` 的段时长先 `round(diff, 4)`，关键耗时再 `int((d or 0) * 1000)` 截断；`round(..., 3)` 与 `round()` 毫秒两处都有独立变异。
- **丢弃账口径**：`queued_chunks_discarded = qsize()` 是**全队列长度**（含其它 turn），而 `_drain` 只丢 `item[1] == turn_id` 的项、保留其它 turn；`drained / kept / remaining_queue` 必须全部暴露，否则「全丢」变异会靠 `qsize` 蒙混过关。
- `chunk_samples = int(sample_rate * chunk_ms / 1000)`，默认 `22050 * 20 / 1000 = 441`。
- 浮点坑：`now=0.15, last=0.0, debounce_ms=150` 时 `int(0.15 * 1000) == 149`（二进制表示），手算锚点改用可精确表示的 `now=0.25, last=0.0, debounce_ms=400`，期望 `since_ms=250`。

## 测试设计

- 7 个 `check_*`：`transition / turn / floor / interrupt / invariants / latency / playback`，judge 双模式 `run` 与 `submit` 均 `7/7 ACCEPTED`。
- 全网格对拍：`transition` 8 状态 × 8 目标 × 3 floor；`floor` 5 action × 8 状态 × 3 floor × 3 turn；`interrupt` 覆盖 debounce 100/200/500、`since == debounce` 边界、`IDLE` 触发但转移被拒；`invariants` 五级优先级全网格 + INV1/INV2；`playback` stale / floor denied / 完整播放 / 中断丢弃四条路径全网格。
- 中间量暴露：`history_len`、`since_ms`、`record`、`violations`、`kept`、`remaining_queue`、`durations_raw` 全部进返回值；否则 `parent`、`since` 缩放之类变异在最终结果上不可观测。
- 手算锚点必须唯一：每条锚点先 `SRC.count(old)` 校验出现次数，避免静默替换错位置（本次 `can-user-floor` 就是踩了这个坑，见下）。

## mutation 设计（40 变异全 CAUGHT）

- 转移族：no-exit 条件 / 拒绝原因 / 删 4 条关键 `VALID` 对 / `FLOOR_BLOCKS` 置空或改 AGENT / 接受后不更新 state / history 不自增；
- floor 族：`STOPPED/ERROR` 漏判、`USER` 持 floor 误放行、turn_id 校验反转、`request_agent_floor` 多传 turn_id、floor 值写反、`release/handle` 与 `should_commit` 分支；
- interrupt 族：`<`→`<=`、`since*1000`→`*100`、`last_trigger_ts` 不更新、连锁时 floor 写错、去抖分支字段；
- invariants 族：shutdown 原因、stale 文本反序、`USER` floor 分支、INV1 文本 / 条件、INV2 条件；
- latency 族：`round(4)`→`round(3)`、`int(*1000)`→`round(*1000)`、marks 白名单取反；
- playback 族：`chunk_samples` 缩放、stale guard 反转、floor denied 反转、`_drain` 丢错 turn、`qsize` 差一、`played_subchunks` 步长、中断位置偏移、`complete` 写反。
- **锚点不唯一踩坑**：`if floor == "USER":` 在源码出现两次（`_can_agent_speak` 的 `if`，以及 `invariants` 里 `elif` 的子串），`SRC.count` 得 2 → 脚本报 `ANCHOR-ERROR can-user-floor expected 1 found 2` 并把它误算为 SURVIVED（39/40）。把锚点扩成带上下文的三行片段后重跑 **40/40 CAUGHT, 0 SURVIVED**。这是脚本 bug，不是语义变异存活。
- 真等价剔除：无（本轮 40 条全部可观测）。

## 查重划界

- 与 `omni-moshi-multistream-delay`（声学 delay 覆盖链）、`omni-thinker-talker-replay`（talker 回放）、`inf-ttft-tpot-queue`（排队时延账）、`asr-moonshine-stream-verify`（流式 ASR 校验）均不同：本题考**全双工 barge-in 控制面 + turn_id stale suppression + 已合成 chunk 丢弃账**。
- agentic 项目步骤的 ReAct / 记忆账本 / 重试熔断是另一层叙事，不是状态机控制面；本题不复用其题面。
- 不考真实音频 / VAD / LLM / TTS / 线程调度；`chunk_samples` 只作确定性整数账。
