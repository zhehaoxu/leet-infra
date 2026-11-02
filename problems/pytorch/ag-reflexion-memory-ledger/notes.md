# notes（ag-reflexion-memory-ledger）

## 回源

- 官方仓 **noahshinn/reflexion**（main 分支；默认分支是 main 不是 master）。记忆流水线三处：`alfworld_runs/env_history.py`（EnvironmentHistory）、`alfworld_runs/generate_reflections.py`（_get_scenario/_generate_reflection_query/update_memory）、`alfworld_runs/webshop_runs/webshop_trial.py`（trial 主循环的滑窗截断）。hotpotqa 侧（agents.py 的 reflections_str 装配 + truncate_scratchpad）依赖 tiktoken，不入题。
- **todos 候选记载纠错**：原记"episodic 滑窗 max 5 条"有误——官方代码全仓无 5 条窗口，webshop/alfworld 的构造期截断与 update_memory 的 query 截断**都是 3 条**（`[-3:]` if `len > 3`）。已按源码钉死。
- **两份 env_history.py 不一致**：webshop 版 `_get_base_query` 段头 `'\nYour memory...'`（单换行），alfworld 版 `'\n\n...'`（双换行）；alfworld 版多一行 `# NOT CURRENTLY SUPPORTED` 注释。以 alfworld（论文主战场）为准，webshop 差异记 description。

## 关键官方 quirk（逐行核实）

1. **滑窗是 3 不是 5**（两处独立代码：构造期 + 反思 query），len==3 恰不截断；构造期截断后 Trial 编号从 0 重数。
2. **add 熔断**：`value == self._last_action`——不是相邻去重，是与**最近一次不同 action** 比较（隔 observation 的重复也触发）；只置 True 永不清；**重复分支不更新 _last_action**——但因熔断粘滞此细节不可观测（eq 变异 SURVIVED 已证：dup 时 value==last_action，赋值本就是 no-op）。
3. **reset 只清 history**：`_is_exhausted` 与 `_last_action` 存活——reset 后重复旧 action 仍熔断（手算锚点钉住）。
4. **渲染**：`cur_query + '\n'` 打头；条目间 `'\n'` 连接、末条无换行；human_edit 前缀 `'[human edit]: '`（官方注释 NOT CURRENTLY SUPPORTED 但 add/str 都支持它——活代码）。
5. **trial log 切分**：run_trial 条目格式 `'\n#####\n\n' + body + '\n\n#####\n'`，相邻接缝 `'#####\n' + '\n#####'` 恰成分隔符 `'#####\n\n#####'`；**实测**：parts 数 = env 数（assert 成立的前提）；part0 = `'\n#####\n\n' + body0 + '\n\n'`（带头残段）、中段 `'\n\n' + body + '\n\n'`、末段带尾 `'#####\n'` + 汇总行。这些残段全部进入 scenario——单条目 log 的 scenario 会以 `#####` 结尾。
6. **scenario**：`split('Here is the task:')[-1].strip()`——多次 marker 取最后一段；吃进 STATUS: FAIL 尾巴（官方 body 是整段 history str，任务行之后还有轨迹与 STATUS）。
7. **反思门** `not is_success and not skip`；反思结果追加**完整** memory；反思 query 的记忆再截一次 `[-3:]`（两处独立滑窗）。
8. **两套 Trial 格式并存**：base query `f'\nTrial {i}:\n{m.strip()}'`（无 #、有 strip）vs 反思 query `f'Trial #{i}: {m}\n'`（有 #、无 strip）。
9. **webshop 侧 latent bug**：webshop main.py 初始化 env_configs 无 `'skip'` 键，而 update_memory 引用 `env['skip']`——alfworld 有。本题缺键转 ValueError。
10. 官方 `env['memory'] += [reflection]` 原地改 env_configs 字典；solve 改为构造新 dict 返回（账面等价）。

## 出题工程

- 输入侧：`ops` 支持 `("reset", "")` 显式建模官方 reset；`few_shot`（官方 reflexion_few_shot_examples.txt）与反思长文头常量显式化；反思长文头逐字符照抄官方（含 'Give your plan after "Plan"'）。
- **测试输入合法性陷阱**：`''.split(sep)` 得 `['']` 而非 `[]`——空 log + 0 env 在官方语义下 assert 必炸；样例检查全部改为 1 个 success env + 单条目 log 的合法组合。
- **手算锚点抓出两处自身错误**：① A,A,B,A 用例——第二问期望 False，实际第二个 A 即触发熔断（A==last A），注释与期望全错；② 单条目 log 的 scenario 尾巴——check_example 首版漏了 `#####` 残段。教训：切分残段 quirk 必须逐字符手算。
- oracle 独立实现：类版 EnvironmentHistory（列表攒段 join）vs solution 的逐条 `+=`；格式串用 % 与 f-string 两种写法降低同源错误。
- 变异 30 条实质全 CAUGHT + 3 条等价记 notes：eq-lastaction-uncond（dup 赋值 no-op）、eq-cap-history-thresh-2（len 3 时 `[-3:]` 恰为全表，阈值 >2 与 >3 等价）、eq-queue-eq-check（qi 每次恰好 +1，`>=` 与 `==` 等价）。
- 红线：反思 LLM 调用、few-shot 文件内容、hotpotqa 侧 scratchpad 截断（tiktoken）、webshop 环境交互不入题。

## 后续候选

- hotpotqa 侧 `format_reflections`/`format_last_attempt`/`truncate_scratchpad`（观测行按 token 长度逐个截断至 1600）可作续题（tokenizer 显式入参化）。
- webshop 侧 `webshop_run` 的 `prompt[-(6400-len(init_prompt)):]` 尾窗截断 quirk 可并入 prompt 预算类题目。
