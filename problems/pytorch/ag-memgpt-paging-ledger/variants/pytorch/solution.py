"""
MemGPT 主上下文换页账：把官方 MemGPT（arXiv 2310.08560，仓今为
letta-ai/letta，以 tag 0.1.6 论文同期代码为准）memgpt/agent.py 的
summarize_messages_inplace + trim_messages/prepend_to_messages 与
memgpt/memory.py 的递归 summarize_messages 无状态化。摘要 LLM 与
token 计数全部显式入参。

solve(messages, all_time_message_count, warning_tokens, tok,
      summarize, time)

- messages：主上下文消息列表 [{"role", "content"}, ...]，index 0
  约定为 system（官方只按位置豁免、不查 role；cut 累计循环里
  system **不豁免**，一样参与 tokens_so_far）。
- all_time_message_count：官方 self.messages_total（全程消息总数，
  含已被换页隐藏的）；None 时官方缺省 len(messages) - 1。
- warning_tokens：MESSAGE_SUMMARY_WARNING_TOKENS（官方 0.1.6 为
  7000），显式入参。
- tok：count_tokens(s) -> int 显式入参（官方 tiktoken per model）。
- summarize：摘要 LLM，纯函数 s -> str（输入即官方喂给
  chat/completion 的 summary_input 字符串）。
- time：package_summarize_message 的 get_local_time() 显式化。

summarize_messages_inplace 口径（官方逐行）：
1. **Smart cutoff**：tokens_so_far=0、cutoff=len(messages)-1；
   reversed(messages) 逐条 tokens_so_far += tok(str(m))（str 整个
   dict 的 repr，非 content），**累加后**判
   tokens_so_far >= warning_tokens * 0.2 才 break，否则 cutoff -= 1
   ——从未触发 break 时循环跑满，cutoff 减到 **-1**（Python 负索引
   参与后续切片/定位）。
2. **保尾**：cutoff = min(len(messages) - 3, cutoff)（官方注释
   Always keep the last two messages too——实际保的是最后 3 个位置
   与 cutoff 的较小者，可与 0/-1 相交产生 quirk）。
3. **cutoff 角色移位（官方 bug）**：if messages[cutoff]["role"] ==
   "user"：new_cutoff = cutoff + 1；内层 if messages[new_cutoff]
   ["role"] == "user" 只打印 "ignoring"——**cutoff = new_cutoff 在
   内层 if 之外，移位无条件发生**；仅当 new_cutoff 越界
   （IndexError）才保持原 cutoff。cutoff=-1 时 messages[-1] 命中
   末条（负索引）。
4. **待摘序列**：seq = messages[1:cutoff]——**system（index 0）
   豁免**；cutoff=0 → 空序列；cutoff=-1 → [1:-1]（丢首尾）。
5. **递归摘要 summarize_messages**：summary_input = str(seq)；
   tok(summary_input) > warning_tokens 时
   trunc_ratio = (warning_tokens / 总量) * 0.8（For good measure）、
   cut2 = int(len(seq) * ratio)（向零截断），
   summary_input = str([递归(前 cut2 条)] + 后段)——递归返回**字符串**
   与剩余 dict 混装进一个列表再 str()；递归对每层独立判定。
6. **记账**：remaining = len(messages[cutoff:])；
   hidden = all_time - remaining；summary_message_count = len(seq)；
   summary_message = json.dumps({"type": "system_alert", "message":
   f"Note: prior messages ({hidden} of {all_time} total messages)
   have been hidden from view due to conversation memory
   constraints.\\nThe following is a summary of the previous
   {summary_message_count} messages:\\n {summary}", "time": time})
   （格式逐字符照抄，含 '\\n {summary}' 的空格）。
7. **trim + prepend**：先 trim_messages(cutoff)（= [messages[0]] +
   messages[cutoff:]，保 system），再 prepend 摘要包
   {"role": "user", "content": summary_message} —— 终态
   [messages[0], 摘要] + messages[cutoff:]；
   messages_total_new = all_time + len([摘要])（摘要也算新增）。

返回账：
{"cutoff_pre_min", "cutoff_pre_shift", "cutoff",
 "summarized": [被摘消息的原始下标序（1..cutoff-1）],
 "summary_inputs": [每次摘要 LLM 收到的输入串，按调用序（含递归层）],
 "summary", "summary_message",
 "summary_message_count", "remaining_message_count",
 "hidden_message_count",
 "new_messages", "messages_total_new"}
域约束：messages 为空 → ValueError（官方会 IndexError，转显式）。
"""
import json


def _recursive_summarize(seq, warning_tokens, tok, summarize, calls):
    summary_input = str(seq)
    summary_input_tkns = tok(summary_input)
    if summary_input_tkns > warning_tokens:
        trunc_ratio = (warning_tokens / summary_input_tkns) * 0.8
        cutoff2 = int(len(seq) * trunc_ratio)
        sub = _recursive_summarize(seq[:cutoff2], warning_tokens,
                                   tok, summarize, calls)
        summary_input = str([sub] + seq[cutoff2:])
    calls.append(summary_input)
    return summarize(summary_input)


def solve(messages, all_time_message_count, warning_tokens, tok,
          summarize, time):
    if not messages:
        raise ValueError("empty messages")
    if all_time_message_count is None:
        all_time_message_count = len(messages) - 1

    # ---- smart cutoff（从队尾反向累计，system 不豁免）
    tokens_so_far = 0
    cutoff = len(messages) - 1
    for m in reversed(messages):
        tokens_so_far += tok(str(m))
        if tokens_so_far >= warning_tokens * 0.2:
            break
        cutoff -= 1
    cutoff_pre_min = cutoff
    cutoff = min(len(messages) - 3, cutoff)
    cutoff_pre_shift = cutoff

    # ---- 角色移位（官方 bug：移位无条件发生）
    try:
        if messages[cutoff]["role"] == "user":
            new_cutoff = cutoff + 1
            if messages[new_cutoff]["role"] == "user":
                pass                    # 官方只打印 ignoring，仍移位
            cutoff = new_cutoff
    except IndexError:
        pass

    # ---- 待摘序列与递归摘要
    seq = messages[1:cutoff]            # system 豁免（按位置）
    calls = []
    summary = _recursive_summarize(seq, warning_tokens, tok,
                                   summarize, calls)

    # ---- 记账
    remaining_message_count = len(messages[cutoff:])
    hidden_message_count = all_time_message_count - remaining_message_count
    summary_message_count = len(seq)
    context_message = (
        f"Note: prior messages ({hidden_message_count} of "
        f"{all_time_message_count} total messages) have been hidden "
        f"from view due to conversation memory constraints.\n"
        + f"The following is a summary of the previous "
        f"{summary_message_count} messages:\n {summary}"
    )
    summary_message = json.dumps({"type": "system_alert",
                                  "message": context_message,
                                  "time": time})

    # ---- trim + prepend
    new_messages = ([messages[0]] + [{"role": "user",
                                      "content": summary_message}]
                    + messages[cutoff:])

    return {"cutoff_pre_min": cutoff_pre_min,
            "cutoff_pre_shift": cutoff_pre_shift,
            "cutoff": cutoff,
            "summarized": (list(range(1, cutoff)) if cutoff >= 0
                           else list(range(1, len(messages) - 1))),
            "summary_inputs": calls,
            "summary": summary,
            "summary_message": summary_message,
            "summary_message_count": summary_message_count,
            "remaining_message_count": remaining_message_count,
            "hidden_message_count": hidden_message_count,
            "new_messages": new_messages,
            "messages_total_new": all_time_message_count + 1}
