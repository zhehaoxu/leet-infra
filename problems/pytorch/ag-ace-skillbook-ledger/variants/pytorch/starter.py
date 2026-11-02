def solve(skills, operations):
    # TODO: ACE 技能书的 delta 批次应用账。
    #   skills 是初始条目 [{"id","section","keywords","issue","insight","active",
    #                        "used","helpful","harmful","neutral"}]，operations 是批次。
    #   - 关键词归一化：strip -> lower -> `\s+` 变 `_`，空串丢，按首次出现去重。
    #   - section 只能是 context / harness；别的值折叠成 context，且原 section 名进关键词，
    #     此时 insight 缺省用 issue 兜底；真正的 context 缺 insight 要报错。
    #   - ADD 关键词为空时默认 [section]；同 skill_id 重复 ADD 会覆盖记录但 section 桶再登记一次。
    #   - UPDATE：issue/insight 为 None 表示不改；keywords 为空列表也表示不改，
    #     但非空却全空白要报错；改完 context 条目必须仍有 insight。
    #   - TAG：delta 只按符号取 ±1，0 记 neutral。
    #   - REMOVE 默认软删（active=False），purge 才真删并清 section 桶。
    #   - MARK_USED 只给 active 条目加 used。
    #   - 提示词按 context -> harness 固定顺序渲染，跳过空节，只有 insight 非空才输出该行。
    raise NotImplementedError
