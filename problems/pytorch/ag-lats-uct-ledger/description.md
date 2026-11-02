# LATS MCTS 搜索账（ag-lats-uct-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：724

把 Language Agent Tree Search（LATS，arXiv **2310.04406**）官方实现 [lapisrocks/LanguageAgentTreeSearch](https://github.com/lapisrocks/LanguageAgentTreeSearch) 的 `programming/mcts.py` 主循环无状态化：Node 的 UCT 选择、best_child 平分规则、expansion/simulation/backprop 的 reward 流转、以及官方代码里几处著名的 bug 级 quirk（visits==0 返回 value 而非 inf、循环变量泄漏决定唯一得 reward 的 child、`"Tested passed:"`/`"Tests failed:"` 拼写不一致）逐一记账。LLM 生成与执行器反馈全部显式入参化。

## 你要实现的

`solve(root_solution, iterations, n, n_tests, plans, root_feedback="", root_reflection="")`：按 plans 逐迭代驱动 MCTS，返回每迭代账面、树终态与终局选择。

## 账点（全部来自官方 mcts.py 逐行核对）

1. **UCT 的 visits==0 quirk**：`if self.visits == 0: return self.value`——官方把 `return float('inf')` **注释掉了**！未访问子节点的 UCT 就是其 value（缺省 0），而非传统 MCTS 的 +inf。
2. **UCT 公式**：`value/visits + 1.0*sqrt(ln(parent.visits)/visits)`——对数里是 **parent.visits**（不是常见的 parent.visits+1）；各节点用**各自父**的 visits。
3. **best_child 平分**：`max(children, key=uct)`——Python max 平分取**先扩展**者。
4. **Selection 恒下行到叶子**：`while node.children: node = node.best_child()`——端点必是叶子；因此内层模拟遍历的 children 恰为当轮新加的 n 个（官方 `for child in node.children` 的"重跑全部"写法在此流程下碰不到历史 child）。
5. **Simulation 的消费序**：每个 child 执行时从队列消费一条 `(passing, feedback, reflection, eval)`；非 passing 写入 reflection（覆盖旧值）、passing 置空；`test_feedback` 无条件写入。
6. **reward_internal 计数**：feedback 含 `"Tested passed:"`（官方拼写）时取 `"Tests failed:"` 前段中 `"Tested passed:"` 后的**非空行数**（strip 后非空）除以 n_tests；无标记则 0。
7. **Break 与 update 对象（官方 bug）**：passing 或末轮（`cur_iter == iterations-1`）对当前 child evaluate 后 break——**break 处的 child** 是 update 对象；未 break 则循环变量泄漏的**最后一个 child** 是 update 对象。同批其余新 child 拿不到任何 reward（官方未 update 它们）。
8. **末轮 quirk**：最后一轮对**第一个** child 即 evaluate 并 break——后续 child 不再模拟。
9. **Reward/Backprop**：`reward = reward_internal + reward_real`（eval 通过 → reward_real=1 且 solved）；update 对象先 `update`（visits+1、value+=reward），再沿 parent 链**到 root** 逐个 update。eval 失败也 break（reward_real=0）。
10. **gather_context_from_tree**：对**选择端点**（父节点）沿 parent 收集非空 feedback/reflection，**反转**为最早在前——新 child 自身的反馈不在其中。
11. **终局选择**：solved → best = 触发 solve 的 child 解；否则 `root.best_child_value()`——max by **累计 value**（非均值！），平分取先；root 无 children 时官方 AttributeError，转 ValueError。
12. **域约束**：expansion 零产出（无 child 可模拟）→ ValueError。

## 样例

```python
plans = [{"solutions": ["a", "b"],
          "sims": [(False, "err", "ra", None),
                   (True, "Tested passed:\nok", "", False)]}]
led = solve("root", 2, 2, 1, plans)
# iter0: c1 非通过继续、c2 通过 → evaluate(False) → break
# updated_child == 2（泄漏/断点处），reward_internal == 1.0
# backprop == [0]（c2 先自 update，再 root）
```

## 计分

12 个检查点：泄漏 update 对象、UCT visits==0、平分取先、末轮首 child evaluate、solved 提前退出、gather 反转与端点、累计 value 终局、reward 计数（空行/拼写）、backprop 累积、域约束，以及 120 组随机 MCTS 运行对拍。
