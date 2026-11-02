"""
LATS (Language Agent Tree Search) MCTS 账：把官方 lapisrocks/
LanguageAgentTreeSearch programming/mcts.py 的 run_mcts 主循环
无状态化。LLM/执行器全部显式入参（expansion 解、模拟结果、
evaluate 结果按消费序入队）。

solve(root_solution, iterations, n, n_tests, plans,
      root_feedback="", root_reflection="")

- root_solution：根节点初始解（官方 pass@1 初解）。
- root_feedback / root_reflection：官方首轮执行写入 root 的
  test_feedback / self_reflection（gather_context 会收集它们）。
- iterations：最大 MCTS 迭代数 max_iters。
- n：每迭代 expansion 的新 child 数。
- n_tests：内部测试数 len(tests_i)，用于 reward 归一化。
- plans：每迭代一个描述符，逐迭代消费：
    {"solutions": [str, ...]        # 恰 n 个新 child 解（expansion 序）
     "sims": [(passing, feedback, reflection, eval), ...]}
       模拟结果队列。官方内层循环按 node.children **全列表序**逐个
       执行（含历史 child——官方 bug：每轮重跑所有 children），
       每个 child 执行时从队首消费一条：passing=内部测试通过；
       feedback=执行反馈；reflection=该次执行生成的自我反思
       （非 passing 才写入 child，覆盖旧值；passing 一律置空）；
       eval=真实测试评估（None 视为 False）。
       循环在 passing 为真或本迭代是最后一轮（cur_iter == max_iters-1）
       时对当前 child 执行 evaluate 后 break——break 时正在处理的
       child 即 update 对象；未 break 则循环变量泄漏的**最后一个
       child** 是 update 对象（官方 bug：同批新 child 只有它得
       reward）。eval 仅在 break 轮被消费。

每迭代流程（与官方逐行对应）：
1. Selection：node=root，`while node.children: node=best_child()`。
   best_child = max(children, key=uct)——Python max 平分取**先扩展**者。
   UCT：visits==0 时返回 self.value（官方注释掉 inf！）；否则
   value/visits + w*sqrt(ln(parent.visits)/visits)（w=1.0）。
2. Expansion：n 个新 Node（depth=父+1）依次 append。
3. Simulation：如上。feedback 含 "Tested passed:" 时
   reward_internal = "Tests failed:" 前段中 "Tested passed:" 后段的
   非空行数 / n_tests，否则 0（官方两处字符串拼写不一致）。
4. Reward/Backprop：reward = reward_internal + reward_real（eval
   通过 → reward_real=1 且 solved）。update 对象先 update（visits+1、
   value+=reward），再沿 parent 链逐个 update（root 也 update）。
5. solved 后提前退出全部迭代。

终局：solved → best = 触发 solve 的 child 解；否则
root.best_child_value()——max by **累计 value**（非均值，平分取先）；
root 无 children → 官方 AttributeError，转 ValueError。

返回账：
{"iters": [...], "nodes": [...], "final": {...}}
迭代记录键：{"iter", "selected_path"(节点 id 序，含起点 root),
"new_ids", "simulated"(本迭代实际执行模拟的 child id 序),
"reward_internal", "reward_real", "updated_child",
"backprop"(自 update 对象的 parent 起、沿链到 root 的 id 序),
"solved", "acc_feedback", "acc_reflection"(对**选择端点**做
gather_context_from_tree 的结果，最早在前——含 root 的
feedback/reflection), "update_child_reflection"(update 对象的
reflection 终值)}
节点记账键：{"id", "solution", "parent", "children", "depth",
"visits", "value", "reflection", "test_feedback"}
final 键：{"best_solution", "best_child_id", "solved"}

节点 id：root=0，此后按创建序递增。
域约束（ValueError）：solved 未发生且 root 无 children 的终局选择；
无 child 可 update（expansion 零产出）时官方 NameError 转 ValueError。
"""
import math


def _reward_internal(feedback, n_tests):
    if "Tested passed:" in feedback:
        passed_section = feedback.split("Tests failed:")[0]
        cnt = len([ln for ln in passed_section.split("Tested passed:")[1]
                   .splitlines() if ln.strip() != ""])
        return cnt / n_tests
    return 0


def _gather(node):
    """gather_context_from_tree：沿 parent 收集 feedback/reflection，
    反转为最早在前。"""
    fb, rf = [], []
    cur = node
    while cur is not None:
        if cur["test_feedback"]:
            fb.append(cur["test_feedback"])
        if cur["reflection"]:
            rf.append(cur["reflection"])
        cur = cur["parent_ref"]
    return fb[::-1], rf[::-1]


def solve(root_solution, iterations, n, n_tests, plans,
          root_feedback="", root_reflection=""):
    nodes = [{"id": 0, "solution": root_solution, "parent": None,
              "children": [], "depth": 0, "visits": 0, "value": 0,
              "reflection": root_reflection,
              "test_feedback": root_feedback, "parent_ref": None}]

    def uct(nd, w=1.0):
        if nd["visits"] == 0:
            return nd["value"]
        par = nd["parent_ref"]
        return (nd["value"] / nd["visits"]) + \
            w * math.sqrt(math.log(par["visits"]) / nd["visits"])

    def best_child(nd):
        if not nd["children"]:
            return None
        return max(nd["children"], key=lambda c: uct(c))

    iters = []
    is_solved = False
    solved_child = None

    for cur_iter in range(iterations):
        if is_solved:
            break
        plan = plans[cur_iter] if cur_iter < len(plans) else None
        if plan is None:
            break
        # Selection
        node = nodes[0]
        path = [node["id"]]
        while node["children"]:
            node = best_child(node)
            path.append(node["id"])
        # Expansion
        sims = list(plan["sims"])
        new_ids = []
        for sol in plan["solutions"][:n]:
            child = {"id": len(nodes), "solution": sol,
                     "parent": node["id"], "children": [],
                     "depth": node["depth"] + 1, "visits": 0, "value": 0,
                     "reflection": "", "test_feedback": "",
                     "parent_ref": node}
            node["children"].append(child)
            nodes.append(child)
            new_ids.append(child["id"])
        # Simulation（官方 bug：遍历全部 children；循环变量泄漏）
        simulated = []
        reward_internal = 0.0
        reward_real = 0
        child = node["children"][-1] if node["children"] else None
        if child is None:
            raise ValueError("no child to simulate")
        for c in node["children"]:
            passing, feedback, refl, ev = sims.pop(0)
            simulated.append(c["id"])
            if not passing:
                c["reflection"] = refl
            else:
                c["reflection"] = ""
            c["test_feedback"] = feedback
            reward_internal = _reward_internal(feedback, n_tests)
            if passing or cur_iter == iterations - 1:
                if ev:
                    reward_real = 1
                    is_solved = True
                    solved_child = c
                break
        updated = c   # 官方循环变量：break 时为正在处理的 child，
        #               未 break 时自然停留在最后一个 child
        acc_fb, acc_rf = _gather(node)
        it_rec = {"iter": cur_iter, "selected_path": path,
                  "new_ids": new_ids, "simulated": simulated,
                  "reward_internal": reward_internal,
                  "reward_real": reward_real,
                  "updated_child": updated["id"],
                  "backprop": [], "solved": is_solved,
                  "acc_feedback": acc_fb, "acc_reflection": acc_rf,
                  "update_child_reflection": updated["reflection"]}
        reward = reward_internal + reward_real
        updated["visits"] += 1
        updated["value"] += reward
        temp = updated
        while temp["parent_ref"] is not None:
            temp = temp["parent_ref"]
            temp["visits"] += 1
            temp["value"] += reward
            it_rec["backprop"].append(temp["id"])
        iters.append(it_rec)

    # 终局选择
    if is_solved:
        best = solved_child
    else:
        root = nodes[0]
        if not root["children"]:
            raise ValueError("no children to choose from")
        best = max(root["children"], key=lambda c: c["value"])
    final = {"best_solution": best["solution"],
             "best_child_id": best["id"], "solved": is_solved}
    for nd in nodes:
        nd.pop("parent_ref")
    return {"iters": iters, "nodes": nodes, "final": final}
