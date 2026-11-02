# notes（ag-memgpt-paging-ledger）

## 回源

- 官方仓 cpacker/MemGPT **已 301 重定向到 letta-ai/letta**（curl API 会拿 404/Moved，须 -L 跟随）。论文同期代码以 **tag 0.1.6**（最旧可用 tag，pymemgpt 时代）为准：`memgpt/agent.py`（AgentAsync.step / summarize_messages_inplace / trim_messages / prepend_to_messages）、`memgpt/memory.py`（递归 summarize_messages）、`memgpt/constants.py`（MESSAGE_SUMMARY_WARNING_TOKENS=**7000**）、`memgpt/system.py`（package_summarize_message）。
- 原始抓取时 curl raw.githubusercontent 反复被 SIGKILL，改用 WebFetch 拿原文——注意 WebFetch 返回的代码有缩进被吞的现象（fetched 文本里方法体缩进丢失），**逐行核对时须以结构推断修正**（例如 `cutoff = new_cutoff` 的缩进层级是本题头号考点，不能靠肉眼看 WebFetch 输出定性，需按 Python 语义与多源交叉验证）。

## 关键官方 quirk（逐行核实）

1. **角色移位 bug（头号）**：`cutoff = new_cutoff` 在内层 `if messages[new_cutoff]["role"] == "user"` **之外**——官方打印 "Shifted cutoff ... is still a 'user', ignoring..." 但**移位照常发生**。注释与行为相反，eq 变异 eq-dead-inner-if（删掉内层 dead if）SURVIVED 即为证明。只有 new_cutoff 越界 IndexError 才保持原 cutoff。
2. **smart cutoff 跑满减到 -1**：`>= 0.2*warning` 才 break、`cutoff -= 1` 在 break 之后——从不触发时循环跑满，cutoff = len-1-len = **-1**；-1 参与 Python 负索引：`messages[-1]` 命中末条（若是 user 则移位到 0）、`messages[1:-1]` 丢首尾、`trim_messages(-1)` = `[m0] + messages[-1:]`。len<3 时 cap（len-3）为负，len=1 时 `messages[-2]` 直接 IndexError（try 只包住移位段）→ cutoff 保持 -2。
3. **system 不豁免于累计**：reversed 循环对全部消息累加 tok(str(m))（str 整个 dict 的 repr）——豁免只发生在两处：待摘切片 [1:cutoff] 与 trim 保留 [0]。cutoff=0 时 trim(0) 保留全表 + prepend → **system 在 new_messages 里出现两次**。
4. **递归摘要**：门 `tok(str(seq)) > warning_tokens` 严格大于（恰等锚点 140==140 不触发）；ratio = (th/总量)*0.8；cut2 = int(n*ratio) 向零截断（round 变异在 2.38 处被抓）；递归返回**字符串**与剩余 dict 混装 `str([sub] + rest)`——repr 长度全靠 dict repr 精确核算（实测 str(dict) 基数 = 27 + len(role) + len(content)，出题时手算错了一版 32+，靠打印实测修正）。
5. **记账**：hidden = all_time - remaining；remaining 用切分后长度（负索引 clamp：len=1 时 messages[-2:] 只有 1 条）；messages_total += len(added)（摘要也算新增）。
6. **打包格式**：`"...\n" + f"The following is a summary of the previous {n} messages:\n {summary}"`——`\n {summary}` 有空格；json.dumps 的 {"type": "system_alert", "message", "time"}。
7. step 侧的 memory-pressure 告警（current_total_tokens > WARNING → agent_alerted_about_memory_pressure 置位/summarize 后复位）是布尔单触发，信息量薄，不入题只记 notes。

## 出题工程

- 入参化：tok / summarize（纯函数 s->str）/ time / warning_tokens 全显式；summarize 的输入就是官方喂给 LLM 的 summary_input 字符串（ledger 记录每次调用的输入序，含递归层——调用序是「内层先、外层后」）。
- **锚点全部按 str(dict) repr 精确核算**：写测试前先打印 len(str(m)) 定基数（27+len(role)+len(content)），别用心算公式；本题三处锚点第一版都错在 str 长度基数上（35/39/43 vs 33/40/44），靠实测脚本纠出——教训：repr 长度必须实测，公式推导必错一次。
- 恰等锚点两个：accum_exact_break（末条 str 恰 39=0.2*195，`>=`→`>` 靠 cutoff_pre_min 区分——post-min cutoff 被保尾 cap 掩盖，必须断言 pre_min）与 recursion_exact_gate（140==140 门不触发 + 139 档 cut2=2 分割点）。
- oracle 独立实现：显式闭包递归 + "".join 分段拼串 vs solution 的 f-string；summarized 索引用 range 表达式重写。
- 变异 25 条实质全 CAUGHT + 3 条等价记 notes：eq-min-commute（min 交换律）、eq-dead-inner-if（官方 dead check）、eq-accum-add-flip（加法交换律）。
- 红线：openai 调用/acreate、faiss archival、persistence_manager 持久化、heartbeat/function 循环（free_step）、persona/human char limit 编辑门不入题；CLI /pop 等不入题。

## 后续候选

- CoreMemory 的 persona/human 编辑门（2000 字符 limit 的报错拼装 + edit_replace 精确匹配）可作 Easy 续题。
- step 主循环的 heartbeat/function_failed/token_warning 三态跳输入状态机可并入 agent 循环类题目。
