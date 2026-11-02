# MemoryBank（Zhong et al., 2305.10250）官方 zhongwanjun/MemoryBank-SiliconFriend
# memory_bank/memory_retrieval/forget_memory.py 标量化。
# 记忆库（含逐条记忆的 strength/last_recall_date/memory_id）、当前日期、
# 每条记忆一次 random.random() 抽样值（显式入参，官方为全局随机）——
# 只考遗忘判定、证据化账本与检索后强化账。
# 官方不建模（记 notes）：FAISS 向量检索与 embedding、LLM 日/全局摘要
# 生成、textsplitter 分块、文件 IO。


def _parse_date(s):
    import datetime
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError(f"date must be YYYY-MM-DD: {s!r}")


def forgetting_curve(t, S):
    """官方原文 math.exp(-t / 5*S)——**优先级陷阱**：等于 exp(-t*S/5)，
    强度 S>1 时衰减更快（与论文意图 exp(-t/(5S)) 相反，官方代码为准）。"""
    import math
    return math.exp(-t / 5*S)


_KW = {"cn": {"mem": "时间{date}的对话内容：", "user": "[|用户|]：",
              "ai": "[|AI伴侣|]：",
              "sum": "时间{date}的对话总结为：{content}"},
       "en": {"mem": "Conversation content on {date}:",
              "user": "[|User|]:", "ai": "[|AI|]:",
              "sum": "The summary of the conversation on {date} is: {content}"}}


def solve(bank, now_date, draws, recalled_ids=(), user=None,
          cur_recall_date=None, language="cn"):
    """遗忘 + 强化两段账（initial_load_forget_and_save +
    update_memory_when_searched）。

    - bank: {用户: {"history": {日期: [对话]}}, 可选 "summary": {日期: str|dict}}；
      对话可为 dict（query/response/...）或旧格式 [query, response]（官方
      迁移写回 dict）。
    - draws: 与「遍历序 = 用户插入序 → 日期插入序 → 对话序」一一对应的
      random.random() 抽样值；遗忘判定 draw > retention。
    - recalled_ids/user/cur_recall_date: 检索命中（FAISS 结果显式化）——
      命中且仍存在的记忆 strength+1、last_recall_date=cur_recall_date；
      定位用 memory_id.split('_')[1] 作日期，只改第一个匹配（break）。

    返回 {"bank": 更新后记忆库, "dialogs": 逐条账（retention/forgotten）,
    "docs": 保留记忆+摘要的 page_content, "updates": 强化账}。
    """
    if not isinstance(bank, dict):
        raise ValueError("bank must be a dict")
    _parse_date(now_date)
    if not isinstance(draws, (list, tuple)) or \
            any(isinstance(d, bool) or not isinstance(d, (int, float))
                or d < 0 or d > 1 for d in draws):
        raise ValueError("draws must be a list of numbers in [0, 1]")
    if language not in _KW:
        raise ValueError("language must be 'cn' or 'en'")
    kw = _KW[language]
    draw_iter = iter(draws)

    def next_draw():
        for d in draw_iter:
            return d
        raise ValueError("not enough draws for all dialogs")

    import copy
    out = {}
    dialogs = []
    docs = []
    for user_name, user_memory in bank.items():
        if not isinstance(user_memory, dict) or \
                "history" not in user_memory.keys():
            continue  # 官方：无 history 的用户整体不进结果库
        um = copy.deepcopy(user_memory)
        out[user_name] = um
        history = um["history"]
        if not isinstance(history, dict):
            raise ValueError("history must be a dict")
        for date, content in list(history.items()):
            _parse_date(date)
            memory_str = kw["mem"].format(date=date)
            user_kw, ai_kw = kw["user"], kw["ai"]
            forget_ids = []
            for i, dialog in enumerate(content):
                if not isinstance(dialog, dict):
                    # 官方旧格式迁移：写回 dict
                    dialog = {"query": dialog[0], "response": dialog[1]}
                    um["history"][date][i] = dialog
                query = dialog["query"]
                response = dialog["response"]
                memory_strength = dialog.get("memory_strength", 1)
                last_recall_date = dialog.get("last_recall_date", date)
                memory_id = dialog.get("memory_id", f"{user_name}_{date}_{i}")
                tmp_str = memory_str
                tmp_str += f"{user_kw} {query.strip()}; "
                tmp_str += f"{ai_kw} {response.strip()}"
                metadata = {"memory_strength": memory_strength,
                            "memory_id": memory_id,
                            "last_recall_date": last_recall_date,
                            "source": memory_id}
                um["history"][date][i].update(metadata)
                days_diff = (_parse_date(now_date)
                             - _parse_date(last_recall_date)).days
                retention = forgetting_curve(days_diff, memory_strength)
                draw = next_draw()
                forgotten = draw > retention
                dialogs.append({"user": user_name, "date": date, "index": i,
                                "memory_id": memory_id,
                                "strength": memory_strength,
                                "days_diff": days_diff,
                                "retention": retention, "draw": draw,
                                "forgotten": forgotten})
                if forgotten:
                    forget_ids.append(i)
                else:
                    docs.append(tmp_str)
            if len(forget_ids) > 0:
                forget_ids.sort(reverse=True)
                for idd in forget_ids:
                    um["history"][date].pop(idd)
            if len(um["history"][date]) == 0:
                # 官方：history 与 summary 同步 pop（缺 key 即崩——本题
                # 改 ValueError，差异记 notes）
                um["history"].pop(date)
                if "summary" not in um.keys() or \
                        date not in um["summary"].keys():
                    raise ValueError(
                        f"summary missing for emptied date {date}")
                um["summary"].pop(date)
        if "summary" in um.keys():
            for date in list(um["summary"].keys()):
                if not isinstance(um["summary"][date], dict):
                    um["summary"][date] = {"content": um["summary"][date]}
                summary_str = um["summary"][date]["content"] \
                    if isinstance(um["summary"][date], dict) \
                    else um["summary"][date]
                metadata = {"memory_strength":
                            um["summary"][date].get("memory_strength", 1),
                            "memory_id": f"{user_name}_{date}_summary",
                            "last_recall_date":
                            um["summary"][date].get("last_recall_date", date),
                            "source": f"{user_name}_{date}_summary"}
                if isinstance(um["summary"][date], dict):
                    um["summary"][date].update(metadata)
                docs.append(kw["sum"].format(date=date, content=summary_str))

    # update_memory_when_searched：命中且仍存在才强化，只改第一个匹配
    updates = []
    if recalled_ids:
        if user is None or cur_recall_date is None:
            raise ValueError(
                "user and cur_recall_date are required for recalled_ids")
        if user not in out:
            raise ValueError(f"unknown user for recall: {user!r}")
        for rid in recalled_ids:
            if not isinstance(rid, str) or len(rid.split("_")) < 2:
                raise ValueError(f"malformed memory id: {rid!r}")
            recalled_date = rid.split("_")[1]
            entry = {"memory_id": rid, "matched": False}
            # 官方直接下标定位（日期不存在即 KeyError——本题改 ValueError）
            if recalled_date not in out[user]["history"]:
                raise ValueError(
                    f"recalled date not in history: {recalled_date!r}")
            for memory in out[user]["history"][recalled_date]:
                if memory["memory_id"] == rid:
                    entry["matched"] = True
                    entry["strength_after"] = \
                        memory["memory_strength"] + 1
                    memory["memory_strength"] += 1
                    memory["last_recall_date"] = cur_recall_date
                    break
            updates.append(entry)
    return {"bank": out, "dialogs": dialogs, "docs": docs, "updates": updates}
