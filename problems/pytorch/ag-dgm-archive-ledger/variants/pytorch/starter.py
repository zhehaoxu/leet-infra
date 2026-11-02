def solve(commits, children, method="score_child_prop", size=2,
          update="keep_all", noise_leeway=0.1, num_issues=60,
          num_full_eval=500, parent_picks=None, pad_picks=None, draws=None,
          baseline=None):
    # TODO: DGM 一轮存档进化的账。
    #   commits 是存档顺序（含 "initial"），children 是本代子代；两者每条都是
    #   {"id", "score", "resolved", "unresolved", "empty", "submitted",
    #    "parent", "metadata_ok", "context_exceeded"}。
    #   - 候选表只收 metadata_ok 的提交；children_count 数的是「存档内父代链接」，
    #     父代不在候选表里就不计数。
    #   - 父代概率：sigmoid(10*(score-0.5))，score_child_prop 再除以 (1+children_count)。
    #   - `best` 按分数**升序**取前 size 个（照抄官方：等于取最差）；名额不足用 pad_picks 补。
    #   - 每个父代按序过三道 25% 闸门（空补丁占比 >= 0.1*total / 随机 / 超长），
    #     硬币从 draws 顺序取，短路的分支不取；未命中就抽一个未解决项，没有就跳过。
    #   - 子代需 metadata_ok、resolved+unresolved>0、submitted>=num_issues 才编译通过。
    #   - 满评门槛：池 = 初始分 + 满评（submitted>=0.9*num_full_eval）成员分，
    #     取降序第二高（不足两个取最高），再与 0.4 取大。
    #   - keep_better 只收分数 >= 初始分-noise_leeway 的编译通过子代。
    raise NotImplementedError
