# notes（ag-dgm-archive-ledger）

## 回源

- 官方实现 `jennyzzt/dgm`：`DGM_outer.py`（337 行）+ `utils/evo_utils.py`（127 行）。
  - `choose_selfimproves`：先遍历 `archive` 建候选表（`accuracy_score` / 三个 id 列表 /
    `children_count`），父代不在表里时 `candidates[parent_commit]` 抛 `KeyError`，被外层
    `except` 吞掉——注意 `candidates[commit]` 已经赋值，所以子代**仍是候选**，只是不计数。
  - 四种父代策略：`no_darwin` 取 `commits[-1:]`；`score_prop` 用
    `1/(1+exp(-10*(score-0.5)))` 归一化；`score_child_prop` 再乘 `1/(1+children_count)`；
    `best` 是 `sorted(..., key=accuracy_score)[:min(size, len)]`——**升序**，取到最差的那几个，
    名额不足用 `random.choices(parent_commits, k=...)` 补。
  - 条目分派：`len(empty_ids) >= 0.1*num_total` 且 `random.random() < 0.25` → `solve_empty_patches`；
    否则 `random.random() < 0.25` → `solve_stochasticity`；否则
    `any_exceeding_context_length(...) and random.random() < 0.25` → `solve_contextlength`；
    都没中且 `unresolved_ids == 0` → `continue`（不产出条目），否则 `random.choice(unresolved_ids)`。
    `and` 短路意味着占比不够时那枚硬币根本不抛——这是本题最容易被忽略的一层。
  - `filter_compiled` → `is_compiled_self_improve`：要求三个 id 键齐全、
    `resolved + unresolved > 0`、`total_submitted_instances >= num_swe_issues[0]`（**只读第 0 个**，
    传 `[small, medium]` 时第二个元素被忽略）。
  - `update_archive(keep_better)`：门槛 `get_original_score() - noise_leeway`，`score >= 门槛` 才入档。
  - `get_full_eval_threshold`：先把初始分无条件放进池子，再遍历 `archive`（**包含 `initial`**）
    把 `submitted >= num_full_eval*0.9` 的分数也放进去 → `initial` 会被数两次；
    取 `sorted(desc)[1]`，只有一项时取 `[0]`，最后 `max(threshold, 0.4)`。
- 论文：arXiv 2505.22954（Darwin Gödel Machine）。论文只给流程，细节一律以官方 `DGM_outer.py` 为准。

## 本题的规范化

- **随机段全部显式化**：父代抽样改成 `parent_picks` 下标序列（概率仍作为账本输出），`best` 的
  名额补齐用 `pad_picks`，闸门硬币用 `draws` 顺序流，未解决项抽取用 `int(coin * n)`。
- **两遍建表**：官方单遍遍历，父代晚于子代出现时那条链接会因 KeyError 丢掉；本题改成「先建全部
  候选、再遍历链接」，于是链接数**与顺序无关**。这一条与原实现的差异已在此明示，测试也不再断言
  单遍行为。
- **`random.choices`** 的精确随机序列不要求复现，只要求「候选顺序 + 权重 + 显式的 picks 下标」
  可判；因此概率是账本字段而非判据本身。
- **`entry_ids` 用条数 + 下标**表示：官方抽的是题目 id，本题抽下标（`int(coin*n)` 归一化，
  与 `random.choice` 的 `randbelow` 不等价，但确定性）。
- `num_swe_issues[0]` 只读首个元素这条不做成陷阱（`num_issues` 直接是标量），但记在此处备查。
- 墙钟 / 线程池 / Docker / 超时全部剔除：只留池管理与判定。

## 变异

34 条全部 CAUGHT（无等价变异）：

```
m-sig-scale            sigmoid 少乘 10              → 概率比值抓
m-sig-shift            sigmoid 不减 0.5             → 概率比值抓
m-prop-no-norm         概率不归一化                 → 和为一抓
m-prop-no-child        score_child 不除子代数       → 双父代用例抓
m-best-desc            best 改成降序                → 取值用例抓
m-no-darwin-first      no_darwin 取第一条           → 基线用例抓
m-gate-ratio-half      占比门槛 0.1 -> 0.5          → 闸门用例抓
m-gate-ratio-strict    占比 `>=` 改成 `>`           → 等号边界抓
m-gate-empty-threshold 第一道阈值 0.25 -> 0.5       → 闸门用例抓
m-gate-ctx-always      第三道不再看 context_exceeded → 硬币消耗抓
m-gate-unresolved-neg  跳过条件写成 < 0             → 跳过用例抓
m-entry-index-shift    下标加一                     → 下标用例抓
m-entry-index-ceil     下标改成 coin*(n-1)          → 下标用例抓
m-compiled-resolved-only  只看 resolved 不看 unresolved → 过滤用例抓
m-compiled-issues-le   题数用 `<=`                  → 差一题用例抓
m-compiled-meta        不查 metadata                → 过滤用例抓
m-threshold-first      门槛取最高而非第二高         → 门槛用例抓
m-threshold-no-floor   不与 0.4 取大                → 门槛用例抓
m-threshold-no-double  池子不预置初始分             → 双计数用例抓
m-threshold-submitted-le  满评判据用 `<=`           → 449/450 用例抓
m-keep-all-always      永远走 keep_all              → 入档用例抓
m-keep-gate-strict     入档 `>=` 改成 `>`           → 等号边界抓
m-keep-gate-leeway     门槛改成加余量               → 入档用例抓
m-keep-uncompiled      未编译子代也入档             → 过滤用例抓
m-cand-meta            不合格 submission 也进候选表 → 计数用例抓
m-count-no-parent-check  父代不查直接自增           → 计数用例抓
m-int-bool             整数校验放过 bool            → validation 抓
m-real-bool            实数校验放过 bool            → 标量边界抓
m-real-finite          不校验有限性                 → validation 抓
m-draws-range          硬币不查到 <1                → 标量边界抓
m-size-min             size 允许 0                  → 标量边界抓
m-method-check         不校验 method                → validation 抓
m-picks-len            不校验 picks 数量            → validation 抓
m-draws-exhaust        硬币耗尽不报错               → 耗尽用例抓
```

## 已知坑

- 先按「每父代 3 枚硬币」写用例，全部多算一枚：`empty` 占比不够时第一道根本不抛，实际是 2 枚。
  用例里的 `draws_used` 断言就是靠这个抓出来的。
- 首轮 `m-keep-gate-strict` 存活：`0.30 - 0.1` 在浮点下是 `0.19999999999999998`，子代 0.20 用
  `>` 也能过。改成 `0.50 - 0.25 = 0.25` 这种**二进制精确**的组合才能区分 `>=` 与 `>`。
- 首轮 `m-draws-range` 存活：`draws=[1.0]` 在变异体里会因为「硬币不够用」抛的同样是 `ValueError`，
  必须给足两枚硬币才能把「取值越界」与「不够用」区分开。
- 首轮 `m-size-min` 存活：`size=0` 配 `parent_picks=[0]` 会先撞上「数量不符」，得配空 picks。
- 「子代排在父代之前」那条最初写了断言，改成两遍建表后行为变了，直接删掉该用例——不留与实现
  不符的期望。

## 查重

- `ag-funsearch-island-ledger`(748)：岛屿程序库的聚类 / 温度采样 / 弱岛重置。本题是**单条存档线**
  上的父代选择与入档判定，没有簇、岛屿与克隆重置。
- `ag-map-elites-archive`(745)：网格精英档案，按行为描述子分格；本题按提交的父子关系组织。
- `ag-reflexion-memory-ledger`(725)、`ag-memorybank-forget-ledger`(722)：反思与遗忘曲线，不涉存档准入。
- `ag-lats-uct-ledger`(724)、`ag-tot-bfs-ledger`：树 / 图上搜索的选择规则，没有「自我修改 + 基准分
  比较 + 编译过滤」三段式。
- `ag-self-instruct-filter`(747)：指令池的过滤与去重，无父子谱系与入档门槛。
