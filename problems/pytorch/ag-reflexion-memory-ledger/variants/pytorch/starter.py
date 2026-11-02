"""
Reflexion 经验记忆账。

把官方 noahshinn/reflexion（arXiv 2303.11366）alfworld_runs/
env_history.py 的 EnvironmentHistory + _get_base_query 与
alfworld_runs/generate_reflections.py 的 _generate_reflection_query
+ update_memory 无状态化。LLM 反思输出全部显式入参。

solve(base_query, start_info, memory, ops, few_shot,
      env_configs, trial_log, reflections)

- base_query / start_info：任务前缀与任务信息。
- memory：当前经验记忆；len > 3 时官方先截 memory[-3:] 再装配
  base query（Trial 编号按截断后列表从头数）。
- ops：EnvironmentHistory 逐条操作 (label, value)：
  ("action", v) / ("observation", v) / ("human_edit", v) /
  ("reset", "")。action 与 _last_action 相同即置 is_exhausted
  （隔 observation 的重复也算，只置 True 永不清）；reset 只清
  history，不清 is_exhausted 与 _last_action。
- few_shot：反思 few-shot 常量（显式入参）。
- env_configs：每 env {"name", "memory", "is_success", "skip"}。
- trial_log：run_trial 写出的 trial 日志全文（按 '#####\\n\\n#####'
  切分，env_logs[i] 对应 env i，条数须等于 len(env_configs)）。
- reflections：反思输出队列，按被反思 env 顺序逐个消费。

update_memory：仅 not is_success 且 not skip 的 env 反思；反思
query 的记忆再截断一次（len>3 → [-3:]）；反思结果追加回完整
memory。scenario = env_logs[i].split("Here is the task:")[-1]
.strip()（多次出现取最后一段）。

返回 {"cur_query", "history_str", "is_exhausted", "queries", "envs"}
——history_str = cur_query + '\\n' + 条目渲染（action → '> '+v、
observation → v、human_edit → '[human edit]: '+v，'\\n' 连接、末条
后无换行）；envs 为反思后的 env 记账（反思过 memory+1 条）。
域约束（ValueError）：非法 label；env 缺键；env_logs 条数不匹配；
reflections 队列不足。
"""


def solve(base_query, start_info, memory, ops, few_shot,
          env_configs, trial_log, reflections):
    raise NotImplementedError
