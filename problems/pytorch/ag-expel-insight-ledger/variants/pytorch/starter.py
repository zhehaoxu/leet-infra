def solve(rounds, initial_rules=None, max_num_rules=10):
    # TODO: ExpeL 的洞见账。
    #   rounds 是逐批的 LLM 复盘输出（原始文本），initial_rules 是 [(规则文本, 计数)]。
    #   - 解析：`(REMOVE|EDIT|ADD|AGREE)( 数字)?: [标签: ]正文`，再过滤「空 / 含 ADD·AGREE·EDIT /
    #     不以句号结尾」；ADD 后面跟的数字丢弃。
    #   - 修复一遍：ADD 的新正文若**包含**某条已有规则 -> 丢；EDIT 的新正文若包含已有规则 -> 改判成
    #     `AGREE <首个匹配序号>` 且沿用旧文本；EDIT 无编号或编号越界 -> 丢；REMOVE/AGREE 匹配不到 -> 丢。
    #   - 落账顺序 REMOVE -> AGREE -> EDIT -> ADD：REMOVE 减 1（条数 >= max_num_rules+5 时减 3）、
    #     AGREE 加 1、EDIT 按**编号**换文本并加 1、ADD 追加且初始计数 2。
    #   - 收尾：丢掉计数 <= 0 的规则，再按计数降序**稳定**排序。
    #   - prompt：`"<序号>. <文本>"` 逐行拼接，序号从 1 起，空账为空串。
    raise NotImplementedError
