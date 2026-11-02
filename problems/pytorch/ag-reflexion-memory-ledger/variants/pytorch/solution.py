"""
Reflexion 经验记忆账：把官方 noahshinn/reflexion（arXiv 2303.11366）
alfworld_runs/env_history.py 的 EnvironmentHistory + _get_base_query
与 alfworld_runs/generate_reflections.py 的 _generate_reflection_query
+ update_memory 无状态化。LLM 反思输出全部显式入参。

solve(base_query, start_info, memory, ops, few_shot, env_configs,
      trial_log, reflections)

- base_query / start_info：任务前缀与任务信息（官方 trial 主循环
  构造 EnvironmentHistory 的前两参）。
- memory：当前经验记忆条目（官方 run_trial 里即 env_config['memory']）。
  **构造 EnvironmentHistory 前官方先截断：len(memory) > 3 时只取
  memory[-3:]（webshop/alfworld 两处一致）**；base query 的 Trial 编号
  0..k-1 按截断后的列表从头数。
- ops：EnvironmentHistory.add 逐条操作 (label, value)：
    ("action", v) / ("observation", v) / ("human_edit", v)
    ("reset", "")  —— 官方 EnvironmentHistory.reset()
  官方 quirk：add 只认三种 label（官方 assert → ValueError）；
  action 与 _last_action 相同即置 is_exhausted=True（非相邻也算——
  中间隔 observation 后重复同样触发），且只置 True 永不清除；
  **reset 只清 history，不清 is_exhausted 也不清 _last_action**。
- few_shot：官方 reflexion_few_shot_examples.txt 常量，显式入参。
- env_configs：每 env {"name", "memory", "is_success", "skip"}（四键
  必备；官方 webshop 侧 config 缺 'skip' 是 latent KeyError bug，
  这里缺键转 ValueError）。
- trial_log：官方 run_trial 写出的 trial 日志全文。
- reflections：反思输出队列，按被反思 env 的顺序逐个消费（官方
  每个 not is_success and not skip 的 env 调一次 get_completion）。

update_memory 口径（官方逐行）：
1. env_logs = trial_log.split('#####\\n\\n#####')；官方
   assert len(env_logs) == len(env_configs)（run_trial 每 env 恰写
   一个 '\\n#####\\n\\n...\\n\\n#####\\n' 条目，相邻条目接缝恰成
   分隔符；env_logs[i] 即 env i 的正文，末段尾巴带汇总行）；
   不相等转 ValueError。
2. 对每个 env i：not is_success 且 not skip 才反思；
   反思 query 用的记忆**再截断一次**：len(env['memory']) > 3 时取
   [-3:]（与第 1 步的构造截断是两处独立代码）；反思结果
   += 追加回**完整** memory（不截断）。
3. 反思 query（官方 _generate_reflection_query）：
   {固定长文头} + ":\\n\\n" + few_shot + "\\n\\n" + scenario；
   若 len(memory) > 0 追加 "\\n\\nPlans from past attempts:\\n" 与
   每条 f"Trial #{i}: {m}\\n"（**不 strip**，与 base query 的
   f"\\nTrial {i}:\\n{m.strip()}" 两套格式并存）；末尾追加
   "\\n\\nNew plan:"。
4. scenario = env_logs[i].split("Here is the task:")[-1].strip()
   （取**最后一段**——正文里多次出现时前者被无视）。

返回账：
{"cur_query": _get_base_query 结果（memory 已截断）,
 "history_str": EnvironmentHistory.__str__ 渲染,
 "is_exhausted": bool,
 "queries": [str, ...]      # 按被反思 env 顺序的反思 query
 "envs": [{"name", "memory", "is_success", "skip"}, ...]  # 反思后}
渲染口径：history_str = cur_query + '\\n' + 各条目（action →
'> ' + value，observation → value 原样，human_edit →
'[human edit]: ' + value）以 '\\n' 连接（末条目后不加换行；
reset 后 history 为空时仅剩 cur_query + '\\n'）。
is_exhausted 在 add 与 reset 之后如实返回（reset 不清）。
域约束（ValueError）：非法 label；env_configs 缺键；
len(env_logs) != len(env_configs)；reflections 队列不足。
"""

REFLECT_HEAD = ('You will be given the history of a past experience in '
    'which you were placed in an environment and given a task to complete. '
    'You were unsuccessful in completing the task. Do not summarize your '
    'environment, but rather think about the strategy and path you took '
    'to attempt to complete the task. Devise a concise, new plan of '
    'action that accounts for your mistake with reference to specific '
    'actions that you should have taken. For example, if you tried A and '
    'B but forgot C, then devise a plan to achieve C with '
    'environment-specific actions. You will need this later when you are '
    'solving the same task. Give your plan after "Plan". Here are two '
    'examples')


def _base_query(base_query, start_info, memory):
    """官方 _get_base_query（alfworld 版：memory 段头双换行）。"""
    query = base_query
    if len(memory) > 0:
        query += "\n\nYour memory for the task below:"
        for i, m in enumerate(memory):
            query += f"\nTrial {i}:\n{m.strip()}"
    query += f"\nHere is the task:\n{start_info}"
    return query


def _reflection_query(log_str, memory, few_shot):
    """官方 _generate_reflection_query（few_shot 显式化）。"""
    scenario = log_str.split("Here is the task:")[-1].strip()
    query = REFLECT_HEAD + ":\n\n" + few_shot + "\n\n" + scenario
    if len(memory) > 0:
        query += "\n\nPlans from past attempts:\n"
        for i, m in enumerate(memory):
            query += f"Trial #{i}: {m}\n"
    query += "\n\nNew plan:"
    return query


def solve(base_query, start_info, memory, ops, few_shot,
          env_configs, trial_log, reflections):
    # ---- 阶段一：EnvironmentHistory 构造 + add/reset（run_trial 侧）
    mem_h = memory[-3:] if len(memory) > 3 else memory
    cur_query = _base_query(base_query, start_info, mem_h)
    history = []
    last_action = ""
    exhausted = False
    for label, value in ops:
        if label == "reset":
            history = []                       # 官方 reset：不清另外两态
            continue
        if label not in ("action", "observation", "human_edit"):
            raise ValueError(f"bad label: {label!r}")
        history.append((label, value))
        if label == "action":
            if value == last_action:
                exhausted = True               # 官方只置 True
            else:
                last_action = value
    parts = [cur_query + "\n"]
    for i, (label, value) in enumerate(history):
        if label == "action":
            parts.append(f"> {value}")
        elif label == "observation":
            parts.append(value)
        else:
            parts.append(f"[human edit]: {value}")
        if i != len(history) - 1:
            parts.append("\n")
    history_str = "".join(parts)

    # ---- 阶段二：update_memory（generate_reflections 侧）
    env_logs = trial_log.split("#####\n\n#####")
    if len(env_logs) != len(env_configs):
        raise ValueError("env_logs / env_configs mismatch")
    queries = []
    qi = 0
    envs = []
    for i, env in enumerate(env_configs):
        for k in ("name", "memory", "is_success", "skip"):
            if k not in env:
                raise ValueError(f"env #{i} missing key: {k}")
        if not env["is_success"] and not env["skip"]:
            mem = env["memory"][-3:] if len(env["memory"]) > 3 \
                else env["memory"]
            q = _reflection_query(env_logs[i], mem, few_shot)
            queries.append(q)
            if qi >= len(reflections):
                raise ValueError("reflections queue exhausted")
            env = {"name": env["name"],
                   "memory": list(env["memory"]) + [reflections[qi]],
                   "is_success": env["is_success"], "skip": env["skip"]}
            qi += 1
        else:
            env = {"name": env["name"], "memory": list(env["memory"]),
                   "is_success": env["is_success"], "skip": env["skip"]}
        envs.append(env)
    return {"cur_query": cur_query, "history_str": history_str,
            "is_exhausted": exhausted, "queries": queries, "envs": envs}
