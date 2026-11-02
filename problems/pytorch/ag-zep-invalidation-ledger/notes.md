# notes（ag-zep-invalidation-ledger）

## 回源

- 论文：Zep: A Temporal Knowledge Graph Architecture for Agent Memory
  （arXiv 2501.13956）§2.3 讲双时态模型：`t_valid` / `t_invalid` 是事件时间，
  `t_created` / `t_expired` 是事务时间；新事实进来时把被顶掉的旧事实的
  `t_invalid` 置为新事实的 `t_valid`。
- 实现：`getzep/graphiti` main（2026-11-02 抓取）。逐段对应：
  - `graphiti_core/utils/maintenance/edge_operations.py`
    - `_dedupe_extracted_edge`（905-1089 行）：空候选早退、快速路径、连续下标翻译、
      duplicate 越界只告警、`resolved` 取首个有效 duplicate、时间戳只给新建边。
    - `_apply_temporal_invalidation`（715-745 行）：先落「自述已历史」的 `expired_at`，
      再按候选升序扫第一个更晚的 `valid_at` 顶掉新边，最后交给
      `resolve_edge_contradictions`。
    - `resolve_edge_contradictions`（748-783 行）：两条跳过判据 + 一条失效判据。
    - `_extract_edge_timestamps`（786-831 行）：「已有任一时间戳就整段跳过」的门。
  - `graphiti_core/utils/maintenance/dedup_helpers.py::_normalize_string_exact`
    （40-43 行）：`re.sub(r'[\s]+', ' ', name.lower()).strip()`。
  - `graphiti_core/utils/datetime_utils.py`：`ensure_utc` 把 naive 当 UTC、
    其他时区转 UTC；`utc_now()` 每次返回新时刻。
  - `graphiti_core/edges.py` 的 `EntityEdge` 字段：`expired_at` 的注释就写着
    「datetime of when the node was invalidated」，与 `invalid_at` 并列。

## 官方口径（逐条钉住）

1. **两对时间戳分工**：`invalid_at` 写事件时间（被顶掉那一刻），`expired_at`
   写事务时间（本次入库的 `now`）。样例与 `writes` 都能区分这两者。
2. **空候选在最前面**：`if len(related_edges) == 0 and len(existing_edges) == 0`
   直接返回 `candidates=None`，连快速路径都不进，也**完全不失效**。
3. **快速路径排在空判定之后**：`related_edges`（不含 `existing_edges`）里文本归一
   相同且两端点相同的第一条即命中；返回的 `duplicate_edges` 是**空列表**
   （`return resolved, [], None`），不是 `[i]`；`candidates=None` 意味着跳过矛盾处理。
4. **连续下标**：`related` 段 `0..n-1`，`existing` 段用 `invalidation_idx_offset`
   偏移。两段的越界判据不同——duplicate 只用 `len(related_edges)`，contradicted 用
   `len(related)+len(existing)-1`；负数在两边都算越界。
5. **`resolved` 取首个有效 duplicate**：官方 `for ...: resolved = ...; break`，
   后面的有效 duplicate 只影响 `duplicate_edges` 列表长度。
6. **duplicate 命中即复用旧边的时间戳**：官方注释
   「duplicated edges retain their existing timestamps」，只有
   `resolved_edge.uuid == extracted_edge.uuid` 时才调 `_extract_edge_timestamps`。
7. **抽取门**：`if edge.valid_at is not None or edge.invalid_at is not None: return`
   —— 任一非空就整段跳过，不是逐字段判断。逐字段那层是 `if valid_at:`（空串跳过）。
8. **第 1 步只看 resolved 自己**：`if resolved.invalid_at and not resolved.expired_at`
   才盖 `expired_at = now`。注意这里 `resolved_expired_at` 一旦非空，第 2 步整体跳过。
9. **第 2 步的三处严格性**：
   - 排序 key `(c.valid_at is None, ensure_utc(c.valid_at))` —— `valid_at` 缺失的排**最后**，
     且**稳定**（同值保原序）；
   - 判据 `candidate_valid_at > resolved_valid_at` 是**严格大于**；
   - 命中即 `break`，多个更晚候选只取排序后最早的那个。
10. **排序是原地操作**：`invalidation_candidates.sort(...)` 改的是传入的列表本身，
    所以紧接着的 `resolve_edge_contradictions(resolved_edge, invalidation_candidates)`
    遍历的是**排过序**的列表，`invalidated_edges` 的顺序随之改变。这是本题最容易被
    「先 copy 再排」的实现漏掉的可观测点，账本里用 `sorted_candidate_uuids` 钉住。
11. **第 3 步三条判据**：
    - `edge.invalid_at <= resolved.valid_at` → 跳过（含等号）；
    - `resolved.invalid_at <= edge.valid_at` → 跳过（含等号）；
    - `edge.valid_at < resolved.valid_at` → 失效（严格小于）。
    官方写成 `if A or B: continue / elif C:`，本题写成「两个 if continue + 一个 if」，
    语义等价（A/B 命中即短路）。
12. **`expired_at` 保留原值**：`edge.expired_at = edge.expired_at if ... is not None
    else utc_now()`，已有过期时刻不重写。
13. **重复候选自然去重**：第二次进入时 `edge.invalid_at` 已被写成 `resolved.valid_at`，
    第一条判据含等号必然命中，所以 `invalidated` 不会出现两条同 uuid。
14. **自指候选不会自杀**：`resolved` 自己出现在候选里时，第三条判据要求严格小于，
    同值不成立。
15. **`now` 全函数共用一个值**：官方在 `_apply_temporal_invalidation` 开头取一次
    `utc_now()`，之后所有 `expired_at` 都用它。本题把它显式化成入参 `now`，
    同一调用内所有写入的 `expired_at` 必然相同。

## 关于 B 分支（`resolved.invalid_at <= edge.valid_at`）

这条判据看着像防御性代码，但**确实能单独改变结果**：当 `resolved.valid_at` 与
`resolved.invalid_at` 构成倒置区间（失效早于生效，LLM 抽错时间的真实产物）时，
一条 `valid_at` 落在 `[resolved.invalid_at, resolved.valid_at)` 的候选，
靠 B 跳过而不会被失效；去掉 B 就会被 C 判失效。用例 `check_new_edge_dead_first`
同时覆盖了这一条与它的等号边界。

相应地，第 2 步写下的 `invalid_at`（`= 某更晚候选的 valid_at > resolved.valid_at`）
永远大于 `resolved.valid_at`，所以 B 与「第 2 步的产物」不可能同时生效——两者互斥，
这也是为什么不能只测「被顶掉之后」的场景。

## 与官方实现的偏离

1. **`utc_now()` 显式化**：官方取系统时钟，不可判；本题 `now` 必填，缺省抛
   `ValueError`。
2. **future-tense 归一化未纳入**：`temporal_edge_utils.py` 的
   `normalize_future_temporal_bounds` 会在时间戳抽取后把「will / plans to / by Friday」
   这类断言按 reference time 改写。它属于**抽取侧**的规范化，与失效判定正交，且
   正则口径本身足够单独成题，故本题严格按「解析 + 写入」简化，并在题面写明。
3. **LLM / embedding / attributes 抽取全部出题**：`decisions`、`timestamps` 都是入参。
4. **`edges` 快照与 `writes` 审计流是新增的**：官方只有内存对象与 DB 写入，没有账本；
    加这两项是为了让「哪条边在什么时候被写成了什么」可逐条核对。

## 判分与变异

41 条变异：40 条实质变异全部 CAUGHT，1 条等价
（`m-package-order`：把 `edges` 的构造从 `for uid in order` 换成 `sorted(order)`）。

等价理由：`edges` 是 dict，Python 的 dict 相等比较与插入顺序无关；`order` 这个变量
在 solution 里只被 `edges` 的构造使用（`state` 才是取值来源）。变异体下 60 组随机夹具
的 `random_cross` 对拍与全部硬断言均通过，印证了「顺序不可观测」。

已覆盖的实质分支（与变异名一一对应）：空路径的 `and`/`or`、空路径漏抽时间戳、
快速路径的三个条件（source / target / 文本归一）、快速路径误扫 `existing`、
duplicate 边界用错（`max_idx` 对 `len(related)`）、漏判负数、取首个还是取末个、
contradicted 边界 `>`/`>=`、负数误映射、`elif` 写成 `else`、
并进旧边仍抽时间戳、抽取门、空串判据、两字段写入序、
第 1 步的 `expired is None` 门、第 2 步 `>`/`>=`、`break` 还是继续、
排序方向 / 排序缺失 / None 排前、第 2 步不排序的门、
第 3 步三条判据各自取等号的方向、A 命中后 `continue` 还是 `pass`、B 分支删除、
漏 `resolved.valid` 非空保护、`expired_at` 保留、候选列表别名导致原序被改写、
`field` 后缀、ISO 格式、naive 时间、`now`/`decisions` 的必填门、
`resolved_is_new` 方向、`candidate_uuids` 取原序还是排序后。

## 做题人容易踩的坑

- 把排序写成 `sorted(candidates)`（返回新列表）——`sorted_candidate_uuids` 与
  `invalidated` 的顺序都会跟官方不一致，尤其是「第 1 步已落 expired」时不排序这一点。
- 把第 3 步的 `expired_at` 无条件写成 `now`，覆盖了候选原有的过期时刻。
- 把「快速路径」实现成「只要 fact 相同就复用」，漏掉两端点，或顺手扫了 `existing_edges`。
- 把 duplicate 的越界判据写成两段总长，导致 `existing` 段的下标被当成 duplicate。
- 把第 2 步的 `>` 写成 `>=`，同 `valid_at` 的候选会把新边顶掉。
