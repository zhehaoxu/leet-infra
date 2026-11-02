"""LATS (Language Agent Tree Search) MCTS 账。

把官方 lapisrocks/LanguageAgentTreeSearch programming/mcts.py 的
run_mcts 主循环无状态化。LLM/执行器全部显式入参。

solve(root_solution, iterations, n, n_tests, plans,
      root_feedback="", root_reflection="")

- root_solution：根节点初始解；root_feedback / root_reflection：
  官方首轮执行写入 root 的反馈与自我反思（gather 会收集）。
- iterations：最大迭代数；n：每迭代 expansion 的新 child 数；
  n_tests：内部测试数（reward 归一化分母）。
- plans：每迭代一个描述符：
    {"solutions": [str]*n,
     "sims": [(passing, feedback, reflection, eval), ...]}
  模拟结果队列：内层按 node.children 全列表序逐个消费一条；
  passing=内部测试通过；feedback=反馈串；reflection=该次执行的
  自我反思（非 passing 才写入 child，passing 置空）；eval=真实
  测试评估（None 视为 False，仅在 break 轮消费）。

每迭代：Selection（node=root，while children: node=best_child()，
best_child = max(children, key=uct)，UCT visits==0 返回 value、
否则 value/visits + sqrt(ln(parent.visits)/visits)，平分取先）→
Expansion（n 个新 child，depth=父+1）→ Simulation（feedback 含
"Tested passed:" 时 reward_internal = "Tests failed:" 前段
"Tested passed:" 后段非空行数 / n_tests，否则 0；passing 或
末轮（cur_iter == iterations-1）对当前 child evaluate 后 break；
break 处的 child 即 update 对象，未 break 则循环变量泄漏的
最后一个 child 是 update 对象）→ Reward/Backprop（reward =
reward_internal + reward_real；update 对象先 update，再沿
parent 链到 root 逐个 update）。solved 后提前退出全部迭代。

终局：solved → best = 触发 solve 的 child 解；否则
root.best_child_value()（max by 累计 value，平分取先；root 无
children → ValueError）。

返回 {"iters": [...], "nodes": [...], "final": {...}}；迭代记录
键 {"iter", "selected_path", "new_ids", "simulated",
"reward_internal", "reward_real", "updated_child", "backprop",
"solved", "acc_feedback", "acc_reflection", "update_child_reflection"}
——acc_* 是对选择端点做 gather_context_from_tree 的结果
（沿 parent 收集 feedback/reflection，最早在前）。节点记账键
{"id", "solution", "parent", "children", "depth", "visits",
"value", "reflection", "test_feedback"}。final 键 {"best_solution",
"best_child_id", "solved"}。节点 id：root=0，按创建序递增。
"""
