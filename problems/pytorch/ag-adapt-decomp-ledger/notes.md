# notes（ag-adapt-decomp-ledger）

## 回源

- 仓库：`archiki/ADaPT` main（论文官方实现）。核心文件 `run_alfworld.py`（603 行）：
  `alfworld_run`（412-460 行，执行器循环）、`plan_and_run`（476-561 行，递归分解本体）、
  `plan_to_args`（462-473 行）、`parse_expression`（381-400 行）、`fetch_args`（402-410 行）、
  `fetch_salient_info`（276-303 行）。`TextCraft/env.py` 与 `run_textcraft.py` 是同一算法的
  另一套环境封装，口径一致。
- 论文：ADaPT（arXiv 2311.05772，NAACL 2024 Findings）§3 把算法讲成「executor 先试、planner
  按需分解、递归」，但没有给出深度上限触发时机、AND/OR 短路后的账本、解析失败兜底这些细节；
  这些只能从代码里读，是本题的主要来源。
- 本题只搬控制流：LLM 调用→查表，环境→脚本，`env.reset()` + 重放 checkpoint → 账本里的
  `replayed` 字段。

## 官方 quirk（逐条核实）

1. **按需 = 先执行再判断深度**：`plan_and_run` 里 `if succ or depth >= max_depth: return` 在
   执行器跑完之后，所以 `max_depth=1` 时**一次分解都不会发生**（根仍然是「先试一次」）。
2. **`max_depth` 是「允许分解的层数」而非「执行层数」**：根 depth=1，规划成功后 `depth += 1`
   才递归，所以 `max_depth=2` 时只有根能分解、子节点只能直接执行。
3. **`range(1, max_runs)` 少跑一步**：`alfworld_run` 的循环从 1 开始，`max_runs=20` 实际最多
   19 步；`max_runs=1` 时一步都不跑。
4. **耐心值用 `==`**：`if pat_ctr == max_patience`，且只看 `"Nothing happens."` / `"OK."`
   两种观测；其他观测清零。触发中断的那一步**在 `prompt += ...` 之前 break**，因此不写进
   completion，但它的动作已经进了 `action_history`。
5. **循环自然结束会抹账**：`if not done: success = False; terminate = True` 且 `return 0`
   ——累加过的 `reward` 直接丢掉（返回的是字面量 0）。
6. **`reward` 先置成功再被抹**：`if reward: success = True; terminate = False` 之后如果又跑到
   耐心中断或脚本耗尽，成功会被压回 `False`。
7. **think 动作三态**：观测强制 `"OK."`（所以也计入耐心）、不进 `action_history`、但进
   `prompt` 与 `num_runs`；完成/失败判定用 `action.lower()` 里的子串。
8. **括号子式是「已解析计划」**：`plan_and_run` 对 `isinstance(task, str)` 为假的分支直接
   `plan_steps = task`，不试执行器、不调规划器、`depth` 也不加——所以分组节点与其子节点同深度。
9. **解析失败整体兜底**：`plan_to_args` 的 `try/except` 把 `parse_expression` + `fetch_args`
   的任意异常吞掉，返回 `{'steps': args, 'logic': 'AND'}`。兜底返回的是**原始字符串列表**，
   括号嵌套就此消失；`Step` 编号越界、右括号多一个、没有执行顺序行都会走这条。
10. **同层逻辑词后写覆盖**：`current['logic'] = token` 是赋值，`Step 1 AND Step 2 OR Step 3`
    整层变成 `OR`；词法只认大写 `AND`/`OR`，小写 `or` 直接当无关字符丢掉。
11. **单步解包**：`len(plan_steps['steps']) == 1` 时把唯一元素提上来；字符串补成
    `{'steps': [x]}`，字典保留原样。官方那段 `if 'logic' not in plan_steps.keys():` 的判断是
    写反的（进入分支后 `plan_steps['logic']` 必然 KeyError），实际效果等价于「缺就补 AND」。
12. **`completion` 只拼成功的、但上限失败也拼**：`running_completion += completion` 在
    `if succ` 之外（深度上限那条分支），所以失败的叶子在返回时仍带着自己的文本。
13. **`checkpoint` 的子集判据是死代码**：`if not set(act_history).issubset(action_checkpoint)`
    里两个名字指向同一个 list（叶子成功时已经 `extend` 过），恒真，因此永远不会重复追加。
14. **`fetch_salient_info` 的字符串算术**：要求字面 `Task completed!`（大小写敏感），
    `rstrip('\n>')` 后 `[:-2]` 丢掉最后两行，再跳过尾部 think 对。以 think 对收尾时恰好取到
    最后一次真实动作；以真实动作收尾时**错位**（取到 `OK.` 与 `>` 之类）；第二段起的 act 行
    还带着上一行残留的 `> ` 前缀。
15. **info 只认字面写法**：执行器用小写 `task completed!` 判定成功，提取信息用大写
    `Task completed!` —— 小写完成的叶子 `success=True` 但 `info_prop=""`。
16. **`all-step-last-act` 的实际效果几乎是「永远为空」**：子节点成功分支只处理
    `last-step-last-act`（另一支没有 else），而 `_update_info` 的 all-step 分支只在叶子成功时
    触发，此时 `past_info` 恒为 `""`（见等价变异 1）。
17. **`env.reset()` + 重放**：每个节点进入时都把环境重置并按 `past_action_checkpoint` 重放，
    但重放的动作不计入 `num_runs`；本题用 `replayed` 记录长度。

## 等价变异（有证明）

1. `m-info-always-last`（把 `_update_info` 的 all-step 分支改成恒等于 last-step）**不可观测**：
   all-step 模式下子任务循环里的 `info` 更新被 `if self.mode == "last-step-last-act"` 挡掉，
   于是任何节点收到的 `past_info` 都恒为 `""`；此时 `(past + "\n" + x).lstrip("\n")` 与 `x`
   逐字符相同，根节点也不更新 `info`，唯一例外是根自己就是叶子——那份 `past_info` 同样是 `""`。
   结论：该分支在任何可达路径上都与 last-step 等价。
2. `m-unmatched-paren-ok`（右括号多一个时不 `stack.pop()` 抛错，而是保留当前节点）**不可观测**：
   保留后 `current.setdefault('steps', []).append(closed)` 会把节点自己塞进自己的 `steps`，
   `_fetch_args` 递归到自引用必然 `RecursionError`；官方是 `pop()` 空栈的 `IndexError`。两条
   路径都被 `plan_to_args` 的裸 `except` 吞掉，最终同样退化成「平铺 + AND」的兜底。

## 变异统计

- 实质变异 37 条全 CAUGHT，覆盖：深度上限时机（`>=`→`>`）、`range(1, max_runs)` 少一步、
  耐心 `==`→`>` 与计数不清零、think 进/不进 history、think 大小写、reward 清 `terminate`、
  自然结束抹 reward/抹成功、completion 分隔符、词法去括号、逻辑词先后、括号回填目标、
  执行顺序行匹配方式（子串 → 行首）、兜底逻辑、单步解包阈值、分组是否加深度、`OR`/`AND`
  短路、`info` 模式与大小写、`[:-2]` 切片、`Task completed!` 剥离、checkpoint 追加、
  `replayed`/`children` 编号、`runs` 计数、`stop` 标签、缺脚本/空计划/模式校验、三个默认值。
- 等价 2 条（见上），均给出结构性证明。

## 工程教训

- **先 trace 再钉断言**：官方 `fetch_salient_info` 的字符串算术（`[:-2]` + 跳过 think 对）
  在「以真实动作收尾」时会错位成 `OK.`/`>` 这种怪值。第一版测试按「最后一次真实动作」
  写期望，直接对不上；把 completion 打出来逐行切才发现口径，最终把「以 think 收尾」和
  「以真实动作收尾」两种形态分别钉住。
- **顶层 `completion` 必带一个尾部换行**：根节点 `running += completion + "\n"`，即使一步
  都没跑（`max_runs=1`）也返回 `"\n"` 而不是 `""`。这类「空值不是空串」的细节只能靠 trace。
- **夹具要能区分「顺手做的」与「规定的」**：'a'/'b'/'c' 如果共用同一份脚本（同一个
  `go to a 1`），checkpoint 会出现三个同值条目，看不出谁写的；改成动作名可区分后断言才有
  信息量。
- **错误用例要避开更早的失败点**：`info_prop_mode="nope"` 若配一份「本来就会缺计划」的输入，
  实现里删掉模式校验也照样抛 `ValueError`（先撞缺计划），变异存活。改成「本来会成功」的
  夹具后立刻捕获。
- **oracle 与参考解要「同语义、异结构」**：oracle 用 `while` 游标 + 递归下降 + 扁平账本，
  参考解用 `_Runner` 类 + 栈式解析；40 组随机剧本交叉验证兜住了两边的实现错。
