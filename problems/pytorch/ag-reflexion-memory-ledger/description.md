# Reflexion 经验记忆账（ag-reflexion-memory-ledger）

**难度**：Medium ｜ **合集**：agentic-systems ｜ **order**：725

把 Reflexion（arXiv **2303.11366**）官方实现 [noahshinn/reflexion](https://github.com/noahshinn/reflexion) 的 alfworld 侧记忆流水线无状态化：`alfworld_runs/env_history.py` 的 `EnvironmentHistory`（base query 三元组装配 + 轨迹渲染 + 重复 action 熔断）与 `alfworld_runs/generate_reflections.py` 的 `_generate_reflection_query` + `update_memory`（trial log 切分、反思门、记忆滑窗与两套 Trial 格式）逐行记账。LLM 反思输出全部显式入参。**注意：官方代码的记忆滑窗是 3 条（`[-3:]` when `len > 3`），不是论文常被转述的 5 条。**

## 你要实现的

`solve(base_query, start_info, memory, ops, few_shot, env_configs, trial_log, reflections)`：先按 run_trial 侧装配 EnvironmentHistory 并逐条 add/reset，再按 update_memory 侧对 trial log 记账，返回装配与反思账。

## 账点（全部来自官方源码逐行核对）

1. **构造期滑窗**：官方 `webshop_run`/`alfworld_run` 在构造 `EnvironmentHistory` 前先 `memory[-3:] if len(memory) > 3 else memory`；base query 的 `Trial {i}` 编号按**截断后**列表从头数（0..2）。len==3 恰不截断。
2. **_get_base_query（alfworld 版）**：`base_query` +（memory 非空时）`"\n\nYour memory for the task below:"` + 每条 `f"\nTrial {i}:\n{m.strip()}"` + `f"\nHere is the task:\n{start_info}"`。webshop 侧同函数是**单**换行段头——两 run 目录的 env_history.py 不一致，本题以 alfworld 为准。
3. **add 的熔断 quirk**：label 只认 `action/observation/human_edit`（官方 assert → ValueError）；action 与 `_last_action` **相同**即置 `is_exhausted=True`——隔 observation 的重复也触发（非相邻比较），熔断只置 True 永不清除；重复分支不更新 `_last_action`（但因熔断粘滞，此细节不可观测——等价变异已证）。
4. **reset 只清 history**：`_is_exhausted` 与 `_last_action` 原样保留——reset 后重复旧 action 仍熔断。
5. **渲染**：`cur_query + "\n"` 后各条目以 `"\n"` 连接、末条后**无换行**；action → `"> " + v`，observation → 原样，human_edit → `"[human edit]: " + v`。
6. **trial log 切分**：`trial_log.split('#####\n\n#####')`——官方 run_trial 每 env 恰写一个 `"\n#####\n\n…\n\n#####\n"` 条目，相邻接缝恰成分隔符，`env_logs[i]` 即 env i 正文；条数不等于 `len(env_configs)` → ValueError（官方 assert）。**切分残段 quirk**：part0 带头 `"#####\n\n"`，末段带尾 `"#####\n"` 与汇总行——它们会进入 scenario。
7. **反思门**：仅 `not is_success and not skip` 的 env 反思；反思结果 `+=` 追加回**完整** memory（不截断）。
8. **反思期二次滑窗**：反思 query 的记忆**再截断一次** `[-3:]` if `len > 3`（与构造期是两处独立代码）；超过 3 条的旧记忆不进 query 但留在 memory 里。
9. **_generate_reflection_query**：固定长文头 + `":\n\n" + few_shot + "\n\n" + scenario`；memory 非空时追加 `"\n\nPlans from past attempts:\n"` 与每条 `f"Trial #{i}: {m}\n"`（**带 #、不 strip**——与 base query 的 `Trial {i}` + `strip()` 两套格式并存）；末尾 `"\n\nNew plan:"`。
10. **scenario 抽取**：`env_logs[i].split("Here is the task:")[-1].strip()`——多次出现取**最后一段**（前段被无视），并吃到 STATUS/##### 尾巴。
11. **消费序**：reflections 队列按被反思 env 的顺序逐个消费；不足 → ValueError。
12. **域约束**：env_configs 缺四键之一（官方 webshop 侧 config 缺 `skip` 是 latent KeyError）→ ValueError。

## 样例

```python
mem = ["a", "b", "c", "d"]                       # len 4 > 3 → 构造期取 ["b","c","d"]
ops = [("action", "A"), ("observation", "o"), ("action", "A")]
cfg = [{"name": "e0", "memory": ["p"], "is_success": False, "skip": False}]
log = "\n#####\n\nEnvironment #0:\nHere is the task:\nT0\n\nSTATUS: FAIL\n\n#####\n"
led = solve("Interact...", "task", mem, ops, "<fs>", cfg, log, ["r"])
# led["cur_query"] 只含 Trial 0..2 = b,c,d（"  a" 被滑窗裁掉）
# led["is_exhausted"] is True（第二个 A == last A，隔 observation 也触发）
# led["queries"][0] 含 "Trial #0: p\n"（反思期记忆只有 1 条，不截）
# led["envs"][0]["memory"] == ["p", "r"]（追加完整 memory）
```

## 计分

12 个检查点：滑窗与 Trial 编号、base query 双格式、熔断（隔步重复/reset 粘滞）、渲染拼接、切分残段、反思门与消费序、二次滑窗、两套 Trial 格式、scenario 末段抽取、三类 ValueError，以及 120 组随机记忆/轨迹/日志对拍。
