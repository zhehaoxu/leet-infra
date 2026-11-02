def solve(skills, events, retrieval_top_k=5, dump_files=None, primitives=None):
    # TODO: Voyager 技能库的写入与检索账。
    #   skills 是已有技能 [{"name","code","description"}]（插入序），events 两种：
    #     {"kind":"add","task","program_name","program_code","llm_description"}
    #     {"kind":"retrieve","query","distances"}   # distances 与当前技能一一对应
    #   - 任务以 `Deposit useless items into the chest at` 开头 -> 整条跳过，记入 skipped。
    #   - 同名技能是**覆盖**：条目数与字典位置都不变，但落盘文件名从 V2 起找第一个空位
    #     （只看 `<name>V<i>.js` 是否已存在）；新技能直接用 name 落盘。
    #   - 描述模板：`async function <name>(bot) {` + `    // <llm 内容>` + `}`。
    #   - 检索条数 k = min(当前技能数, retrieval_top_k)，空库 k=0 且返回空表；
    #     排序按距离升序、并列按插入序。
    #   - programs：先所有技能代码、再控制原语，每段后跟一个空行。
    raise NotImplementedError
