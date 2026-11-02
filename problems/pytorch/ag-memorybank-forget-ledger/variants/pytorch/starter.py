def solve(bank, now_date, draws, recalled_ids=(), user=None,
          cur_recall_date=None, language="cn"):
    """MemoryBank（2305.10250）官方 zhongwanjun/MemoryBank-SiliconFriend
    memory_bank/memory_retrieval/forget_memory.py 标量化账。

    - forgetting_curve(t, S) = math.exp(-t / 5*S)（官方原文优先级陷阱，
      等于 exp(-t*S/5)：S>1 衰减更快，与论文意图 exp(-t/(5S)) 相反）。
    - 遍历序 = 用户插入序 → 日期插入序 → 对话序；旧格式 [q, r] 对话迁移
      写回 dict；缺省 memory_strength=1、last_recall_date=所属日期、
      memory_id=f"{user}_{date}_{i}"（.update 写回条目）。
    - 遗忘判定 draw > retention（draws 与遍历序一一对应）；遗忘下标逆序
      pop；日期清空时从 history 与 summary 同步删除（summary 缺 key 即
      官方 KeyError，本题 ValueError）；无 history 键的用户整体不进结果库。
    - 摘要不参与遗忘：非 dict 包成 {"content": ...} 并写回元数据
      （memory_id=f"{user}_{date}_summary"），产出摘要文档。
    - 检索强化（update_memory_when_searched）：按 id.split('_')[1] 定位
      日期（缺失即官方 KeyError→ValueError），首个匹配 memory_id 命中后
      strength+1、last_recall_date=cur_recall_date 并 break。

    返回 {"bank": 更新后记忆库, "dialogs": 逐条账（days_diff/retention/
    draw/forgotten）, "docs": 保留对话与摘要的 page_content,
    "updates": 强化账}。域约束：日期为 YYYY-MM-DD、draws 为 [0,1] 数值、
    language ∈ {cn, en}、draw 数不足 / 证据行缺失 → ValueError。
    """
    raise NotImplementedError
