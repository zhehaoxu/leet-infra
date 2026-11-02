# notes（ag-lats-uct-ledger）

## 回源

- 官方仓 **lapisrocks/LanguageAgentTreeSearch**（原 andyz245/LanguageAgentTreeSearch；lats-group/lats 404）。MCTS 核心在 `programming/mcts.py`（hotpot/webshop 各有同构副本）。**论文号修正：LATS 是 arXiv 2310.04406**（Zhou et al., ICML 2024）——todos 候选里写的 2407.01476 有误，已同步修正。
- 文件树：`programming/{mcts,dfs,reflexion,immediate_refinement,main}.py` + executors/generators。MCTS 主循环 `run_mcts`（~230 行）全量拿到。

## 关键官方 quirk（逐行核实）

1. **UCT visits==0 返回 value**（`#return float('inf')` 被注释）——未访问节点 uct=value=0。推论：**同层兄弟至多一个被访问**——每迭代只有泄漏/break 的一个 child 得直接 update，backprop 只沿单一路径；未访问兄弟 uct=0，被任何 reward≥0 的已访问兄弟压过（或平分靠先手序）。⇒ **UCT 的 mean-vs-sum 在官方流程内不可观测**（eq 变异实测 SURVIVED，记 notes）——reward 恒 ≥0（计数/n_tests ≥0、real∈{0,1}），均值与累计同号且未访问者恒 0。
2. **循环变量泄漏决定 update 对象**：`for child in node.children: ... break` 之后 `child.update(reward)`——break 时是断点 child，未 break 是最后一个 child；**同批其余 child 零 update**（官方没有对它们 update）。
3. **selection 恒下行到叶子** ⇒ 内层 `for child in node.children` 实际只遍历当轮新加的 n 个（历史 child 不会被重模拟）——"重跑全部 children"的官方写法在此流程下无害。第一版测试误以为会重跑历史 child，被插桩证伪后修正。
4. **末轮 quirk**：`if is_passing_internal or cur_iter == max_iters - 1:` 对**第一个** child 即真 → 只模拟第一个 child 就 evaluate+break。iterations=1 的用例全部走这条路——第一版锚点在 iterations=1 下构造"全部模拟"全挂。
5. **reward 计数**：`"Tested passed:"`（Tested！）与 `"Tests failed:"`（Tests）拼写不一致；`split("Tested passed:")[1]` 取后段、非空判定 `line.strip() != ''`（纯空格行不算）；无 "Tested passed:" 时 reward=0（注意分母 len(tests_i) 是内部测试数）。
6. **gather_context_from_tree**：沿 parent 收集、`[::-1]` 反转（最早在前）；对**选择端点**调用（新 child 的反馈不在内）。root 的首轮 feedback/reflection 也会收集——入参 root_feedback/root_reflection。
7. **终局 best_child_value**：`max(children, key=lambda c: c.value)`——**累计 value 非均值**；root 无 children → None.solution AttributeError → ValueError。
8. **eval 失败也 break**：`if is_passing_internal or last_iter:` 块内 evaluate 后**无条件 break**——eval False 时 reward_real=0 但循环仍终止。
9. **backprop 到 root**：`temp = child; while temp.parent: temp = temp.parent; temp.update(reward)`——child 已在循环外 update 过，链上从 parent 到 root 全部 +1（root 无 parent，也吃到）。

## 出题工程

- 模拟结果四元组 `(passing, feedback, reflection, eval)` 按官方**执行序**入队（每 child 执行消费一条）；eval 仅在 break 轮消费——队列给余量即可。
- **ledger 的 updated_child 语义是本题最大坑**：第一版 solution/oracle 都把 update 对象写成预初始化的 `children[-1]`，break 语义丢失——oracle 与 solve 同错时 _compare 察觉不到，靠**手算锚点**（break_on_old_child）抓出。教训：oracle 与 solve 同源错误必须靠独立手算锚点兜底。
- **oracle 的 gather 忘了 [::-1] 反转**——random 对拍第 13 组抓出（acc_feedback 顺序反）。oracle 不是真理，逐行对照官方源码是唯一标准。
- nodes 终态输出：solve 原地 pop parent_ref（嵌套 children 同对象联动干净）；oracle 须**递归**清洗 pref——只洗顶层漏嵌套，random 抓出 {'pref'} 键差。
- 变异 18 条实质全 CAUGHT + 2 条等价（sum-not-mean、passing-keep-refl——后者官方本就冗余：新 child reflection 缺省 ""）。**eq 变异曾被误判 CAUGHT**：根因是当时 flip 锚点本身期望错误（把两层 selection debug 输出误读成一层、把 mid-run 状态当终态）——修复锚点后 eq 变异回归 SURVIVED。教训：先证明检查对正确实现通过，再用它判变异。
- 红线：prompt 模板、gen.func_impl 的重试循环、exe.evaluate 的沙箱执行、prune_context_blocks（split('Previous Trial')/join('trial') 定界符不一致 bug 很精彩但 context 剪枝在 mcts.py 主循环外未被调用——死代码不入题）、pass@k 统计不入题。

## 后续候选

- hotpot 版 LATS（ReAct-style action 空间 + wikienv）与 webshop 版的 MCTS 变体可作对比题；dfs.py 的 DFS 账（vs MCTS 的选择差异）可并入 ToolLLM DFS 候选一起考量。
