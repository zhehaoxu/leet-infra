# notes（ag-ace-skillbook-ledger）

## 回源

- 官方实现 `kayba-ai/agentic-context-engine`，`ace/core/skillbook.py`（899 行），另核过
  `ace/deduplication/operations.py`（去重探针，本题不用）：
  - `_normalize_optional_text` / `_normalize_required_text`：`str()` + `strip()`，空串成 `None`。
  - `_normalize_keyword`：`strip` 后 `re.sub(r"\s+", "_", text.lower())`；`_normalize_keywords`
    用 `seen` 集合按首次出现去重。
  - `_coerce_section_and_keywords`：section 合法则原样返回；否则 keywords 变成
    `normalize([section] + keywords)`，section 折叠为 `DEFAULT_LEGACY_SECTION = "context"`。
  - `add_skill`：`if not normalized_keywords: normalized_keywords = _normalize_keywords([raw_section])`；
    `if section == "context" and insight is None:` 再分「raw section 不合法 → insight = issue」
    与「合法 → raise」两支；`self._sections.setdefault(section, []).append(skill_id)` 无条件 append，
    所以重复 id 会重复登记。
  - `update_skill`：`issue`/`keywords`/`insight` 用模块级 `_UNSET` 哨兵区分「不改」；keywords 归一化
    后为空则 `raise ValueError("keywords are required and must be non-empty")`；
    之后 `if skill.section == "context" and not skill.insight: raise`。
  - `_apply_operation`：UPDATE 传 `operation.keywords if operation.keywords else _UNSET`
    —— **空列表等价于不改**；TAG 的 `delta = int(metadata.get("delta", 0))` 再按符号夹成 ±1。
  - `tag_skill` / `mark_used` / `remove_skill(soft=True)` / `purge` / `as_prompt` / `stats` 逐行对照。
- 论文：arXiv 2510.04618（Agentic Context Engineering）。论文描述 generator/reflector/curator 三角色，
  但 `helpful/harmful` 计数与渲染顺序这类细节以官方 `skillbook.py` 为准。

## 本题的规范化

- **时钟与 embedding 剔除**：`created_at` / `updated_at` / `embedding` 不进账本（`update_skill`
  会把 embedding 置空，属不可判定的附属状态）。
- **id 生成**：官方用内部 `_generate_id`，本题归一化为 `"<section>-<序号>"`（序号 = 当前条目数 + 1），
  只在不传 `skill_id` 时触发。
- **`insight_source` / `occurrences` / `similarity_decision` 整族剔除**：那是「洞见来源溯源」与
  「两次相似的条目是否合并」的记账，与 delta 应用本体正交，留作后续题目。
- **`apply_update` 的中途异常**：官方不捕获，一条坏操作会让整批中止且前面的操作已经落地。
  本题同样直接抛 `ValueError`（部分落地的状态不可观测，故不建模）。
- **`MARK_USED`** 不是模型产出的 delta 操作，是框架的 `mark_used(skill_ids)` 公开方法；为了让它可判，
  本题把它提升成一种操作类型，并在题面写明。

## 变异

30 条全部 CAUGHT：

```
m-text-no-strip          不 strip                         → 归一化用例抓
m-kw-no-lower            关键词不转小写                   → 归一化用例抓
m-kw-no-ws               `\s+` 不换下划线                → 归一化用例抓
m-kw-no-dedupe           关键词不去重                     → 归一化用例抓
m-legacy-no-fold         遗留 section 不折叠              → 折叠用例抓
m-legacy-no-kw           遗留 section 名不进关键词        → 折叠用例抓
m-legacy-no-insight      遗留 section 不用 issue 兜底     → 折叠用例抓
m-context-no-raise       context 缺 insight 不报错        → 折叠用例抓
m-default-kw             空关键词不补 [section]           → ADD 用例抓
m-update-empty-kw-sets   空列表也当成「改成空」           → UPDATE 用例抓
m-update-insight-unset   空串 insight 不再清空            → 空串用例抓
m-update-no-context-check  update 后不查 context insight  → 清空用例抓
m-tag-delta-raw          delta 按原值累加                 → 计数用例抓
m-tag-zero-helpful       neutral 记成 helpful             → 计数用例抓
m-remove-soft-noop       软删不置 active                  → 软删用例抓
m-purge-not              purge 退化成软删                 → 软删/真删用例抓
m-purge-keep-bucket      purge 不清节列表                 → 节列表断言抓
m-used-any-active        非 active 也加 used              → 计数用例抓
m-used-no-bump           used 不加                        → 计数用例抓
m-prompt-order           两节渲染顺序颠倒                 → 渲染用例抓
m-prompt-empty-section   空节也输出标题                   → 渲染用例抓
m-prompt-insight-always  insight 恒输出（含 None）        → 渲染用例抓
m-prompt-no-strip        末尾不去空行                     → 渲染用例抓
m-stats-total-active     skills 只数 active               → stats 用例抓
m-stats-sections-sum     sections 数节键                  → stats 用例抓
m-add-dup-guard          重复 id 不再重复登记             → 重复 ADD 用例抓
m-op-unknown             未知操作类型放行                 → validation 抓
m-delta-bool             delta 放过 bool                  → validation 抓
m-skill-nonneg           计数不再要求非负                 → validation 抓
m-markused-any-seq       skill_ids 放过字符串             → validation 抓
```

## 已知坑

- 第一版用 `insight: "  "` 想抓「空串清空」那条变异，结果 `"  "` 是**真值**，`if insight:` 也会进
  分支，变异存活；改成真正的 `""` 才区分开「不改」与「清空」。
- `{"type": "NOPE"}` 单条就能「抓到」未知类型，但抓的是别的原因（`else` 分支走 `MARK_USED`，
  `skill_ids` 缺失 → 报错）。要真正验证 kind 白名单，得给一条**字段齐全**的未知操作。
- `stats["sections"]` 与 `stats["by_section"]` 的键数不是一回事：前者只数非零的节，后者恒含两个键。

## 查重

- `ag-memgpt-paging-ledger`（页式记忆换入换出）、`ag-zep-invalidation-ledger`（失效传播）：
  都是「记忆条目的生命周期」，但机制是分页与失效，不是 delta 操作批 + 计数器 + 渲染。
- `ag-reflexion-memory-ledger`(725)：反思条目的写入与检索，没有 `_UNSET` 哨兵、软删与节折叠。
- `ag-self-instruct-filter`(747)：候选指令的过滤去重，条目只增不减。
- `ag-funsearch-island-ledger`(748)、`ag-dgm-archive-ledger`(749)：程序库 / 存档的进化池管理，
  与「策略文档按增量操作改写」正交。
- `peft-trainable-tokens-replace` 只是名字里都有 token/条目，机制无关。
