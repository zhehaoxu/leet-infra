# notes（ag-expel-insight-ledger）

## 回源

官方实现 `LeapLabTHU/ExpeL`，核过两份文件：

- `agent/expel.py`（743 行）
  - `parse_rules`(L665-681)：L666 的正则是
    `((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)`；L673 的三道门
    （非空 + 不含 `['ADD','AGREE','EDIT']` + `endswith('.')`）；L677 `if 'ADD' in operation` 归一化。
  - `update_rules`(L696-743)：L698-719 第一遍只改 `operations`（ADD 去重、EDIT 命中已有规则改判
    `AGREE`、编号越界 / 找不到原文的一律进 `delete_indices`）；L721-739 第二遍按
    `['REMOVE','AGREE','EDIT','ADD']` 固定顺序落账；L730 `remove_strength = 3 if list_full else 1`；
    L740 过滤 `> 0`；L741 `sort(key=count, reverse=True)`。
  - 满额判据 L314：`list_full = self.max_num_rules + 5 <= len(self.rule_items_with_count)`，
    在**应用本批之前**求值。
  - 渲染 L399：`'\n'.join([f"{i}. {item}" for i, item in enumerate(self.rule_items, 1)])`，
    注意渲染的是**纯文本**列表（计数只进日志）。
  - `retrieve_rule_index`(L683-687) / `is_existing_rule`(L689-693)：匹配方向是
    `rules[i][0] in operation_rule_text`，即「旧文本 ⊂ 新文本」，且返回**首个**命中下标。
- `insight_extraction.py`（169 行）：L144 `react_agent.create_rules(...)`，L156 落盘字段白名单。
  critique 的三条来源（失败 >1 才复盘 L177、成功按 `success_critique_num` 分块 L159、L179 的
  13000 token 截断三策略）都在 `expel.py` 侧，与本题的账正交。

## 本题的规范化

- **LLM 输出整体入参化**：`rounds` 就是逐批的 `llm_output` 原文，不再真的调模型；`parse_rules`
  的正则与三道门原样保留，因为它是本题的主要考点之一。
- **轨迹配对与 critique 门整层剔除**：成功 × 失败的交叉、失败次数 > 1 才复盘、`random_divide_list`
  的随机分块、13000 token 截断的 `random` 策略——要么是随机主导，要么要真的 token 计数，都不进账。
  本题只从「已经拿到 LLM 输出」这一步开始。
- **embedding / vectorstore / fewshot 检索剔除**：ExpeL 的 top-k 只发生在 fewshot 轨迹检索上
  （`setup_vectorstore` + `topk_docs`），规则是**全量注入**提示词的（`insert_before_task_prompt`
  直接塞 `self.rules`）。所以「洞见账」这一侧没有任何 kNN，本题也不引入，避免与
  `ag-generative-memory-retrieval` 撞车。
- **断点续跑剔除**：`create_rules` 的 `loaded_dict`/`loaded_log`/`resume_flag` 一整套是给长跑任务
  续命用的，账本身不依赖它。
- **`max_num_rules` 的作用被钉死为「只影响 REMOVE 力度」**：官方从不按它裁剪，题面与提示都写明。

## 变异

29 条，28 条 CAUGHT，1 条实证等价：

```
m-regex-no-label          不吞 `标签: ` 前缀              → parse_prefix_label 抓
m-regex-no-number         操作名后不再允许数字             → parse 各例抓（EDIT 3 直接不匹配）
m-regex-anchored          正则加 ^ 锚定                    → midword 用例抓（PADD 不再命中）
m-banned-empty            禁词表清空                       → three_gates 抓
m-banned-remove           禁词表加上 REMOVE                → three_gates 抓
m-no-period-gate          不要求句号结尾                   → three_gates 抓
m-banned-substring        禁词判定改成整串相等             → three_gates 抓
m-add-number-kept         ADD 保留编号                     → add_parses_into_count_two 抓
m-int-bool                整数校验放过 bool                → validation 抓
m-match-reversed          包含方向反过来                   → add_dedupes_by_substring 抓
m-add-dedupe-inverted     ADD 去重条件取反                 → add_dedupes_by_substring 抓
m-edit-agree-index        改判 AGREE 的序号 +2             → edit_becomes_agree 抓
m-edit-num-off            编号上界放宽一位                 → edit_invalid_number 抓
m-remove-exist-inverted   REMOVE/AGREE 存在性取反          → remove_unknown_text 抓
m-remove-strength-one     REMOVE 永远 -1                   → remove_strength 抓
m-remove-strength-three   REMOVE 永远 -3                   → remove_strength 抓
m-agree-double            AGREE 加 2                       → prune_and_stable_sort 抓
m-edit-no-bump            EDIT 不加计数                    → edit_by_number 抓
m-edit-index-zero         EDIT 永远改第 0 条               → edit_by_number 抓
m-add-count-one           ADD 初始计数 1                   → add_parses_into_count_two 抓
m-order-edit-first        落账顺序改成 EDIT 先             → apply_order 用例抓（见下）
m-prune-zero              计数 0 不再剔除                  → prune_and_stable_sort 抓
m-sort-asc                排序方向反了                     → prune_and_stable_sort 抓
m-full-threshold          满额阈值改成 `max <= len`        → remove_strength 抓
m-prompt-off-by-one       渲染序号从 2 起                  → render_prompt 抓
m-count-log               count 数轮数                     → add_dedupes / render 抓
m-log-applied-from-parsed applied 记成解析结果（未修复）   → edit_becomes_agree 抓
m-listfull-const          list_full 恒 True                → remove_strength 抓
m-min-zero                `value < minimum` → `value < 0`  → 等价（见下）
```

**等价变异的实证**：`_as_int` 在本题只有两处调用，唯一传了 `minimum` 的是 `max_num_rules`，而它传的
就是 `0`。于是 `value < minimum` 与 `value < 0` 在该调用点逐值等价（`-1` 在两边都触发、`0` 在两边
都放行）。这不是漏测：把 `minimum` 参数删掉、直接写 `value < 0` 也能通过全部用例。

## 已知坑

- **句号让「包含」关系变得罕见**：所有规则都以 `.` 结尾，所以「旧文本 ⊂ 新文本」只有在旧规则是
  新规则里完整一句（含句号）时才成立，例如旧 `Check the drawer.` 与新
  `Check the drawer. Then open it.`。第一版去重用例写成旧 `open the drawer.` + 新
  `open the drawer before checking.`，句号把匹配切断，结果「重复」根本没被判出来，用例反而错了。
- `list_full` 需要 `len >= max_num_rules + 5`；`max_num_rules` 最小是 0，所以至少要 5 条规则才能
  触发 -3。混合用例里条数根本到不了 5，只能单独造 5 条初值。
- `m-order-edit-first` 是**崩溃式被抓**：换序后 AGREE 仍按文本找，而 EDIT 已经把该文本换掉，
  `retrieve_rule_index` 返回 `None`，`rules[None]` 直接 `TypeError`。这条正好说明官方那句
  `# Order is important` 不是摆设——顺序被写死是为了让 EDIT 在 AGREE 之后。
- 排序稳定性：`sort(key=count, reverse=True)` **不会**反转同分元素；oracle 里写成
  `sort(key=lambda r: -r[1])`（升序取负），两者对同分元素给出相同顺序。
- `check_against_oracle` 第一版把第二轮期望成「EDIT 3 会落地」，实际是越界丢弃；同一轮里的 ADD 也因为
  「新正文包含旧规则全文」被整条丢掉，整轮空转——这条最后反而成了断言的一部分。

## 查重

- `ag-ace-skillbook-ledger`(750)：也是「LLM 出操作、我们落到一份文本账上」，但那边条目带 id、
  按 id 定位、有关键词归一化与 section 折叠、`delta` 只取符号、软删 / purge；本题条目**没有 id**，
  定位靠「旧文本是新文本的子串」，计数是有符号整数（ADD=2、AGREE=+1、REMOVE=-1/-3、EDIT=+1），
  并按计数修剪与排序。两题的判据没有一行重合。
- `ag-voyager-skill-lib`(751)：技能库的同名覆盖 + 版本化落盘 + top-k 检索，不涉及规则文本匹配与计数。
- `ag-funsearch-island-ledger`(748)、`ag-dgm-archive-ledger`(749)：程序池 / 存档的进化采样，
  条目只增不删、也没有计数修剪。
- `ag-reflexion-memory-ledger`：反思条目的写入与检索，没有操作批次与计数淘汰。
- `ag-memgpt-paging-ledger`、`ag-zep-invalidation-ledger`：页式换入换出与失效传播，机制无关。
