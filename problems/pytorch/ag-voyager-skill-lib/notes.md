# notes（ag-voyager-skill-lib）

## 回源

官方实现 `MineDojo/Voyager`，`voyager/agents/skill.py`（127 行），整份读过并逐条对照：

- **跳过闸门**（L62-64）：`if info["task"].startswith("Deposit useless items into the chest at"): return`
  ——整条 `add_new_skill` 直接返回，不落盘、不进库，也**不产生任何记录**（本题补了一条 `skipped` 记账）。
- **同名覆盖**（L71-79）：`if program_name in self.skills:` → 先
  `self.vectordb._collection.delete(ids=[program_name])`，再
  `i = 2; while f"{program_name}V{i}.js" in os.listdir(...): i += 1`，落盘名 `f"{program_name}V{i}"`；
  否则（L78-79）`dumped_program_name = program_name`。
- **写库顺序**（L80-88）：`add_texts(ids=[program_name])` 之后才 `self.skills[program_name] = {...}`。
  `self.skills` 是 **dict**，同名键赋值保持原插入位置——这就是「覆盖后 `order` 不变」的来源。
- **`synced` 的来源**（L45-50 与 L89-91）：两处 `assert self.vectordb._collection.count() == len(self.skills)`。
  覆盖分支在 L73 先 `delete` 掉同 id 再 `add_texts`，所以 count 不涨——这条断言在任何合法路径上恒真。
- **描述模板**（L102-112）：`skill_description = f"    // {self.llm(messages).content}"`，再
  `return f"async function {program_name}(bot) {{\n{skill_description}\n}}"`。
  注意四空格 + `// ` 两层前缀，题目里合并成一层模板。
- **检索**（L114-127）：`k = min(self.vectordb._collection.count(), self.retrieval_top_k)`，
  `if k == 0: return []`，然后 `similarity_search_with_score(query, k=k)`，按返回顺序取
  `self.skills[doc.metadata["name"]]["code"]`。
- **programs**（L52-59）：先按 `self.skills.items()` 顺序拼 `f"{entry['code']}\n\n"`，再拼所有
  control primitives，同样 `\n\n`。
- 论文 arXiv 2305.16291 描述「技能库 + 自动课程 + 迭代提示」三层，但落盘/覆盖/检索的细节全在
  `skill.py`，以代码为准。

## 本题的规范化

- **向量库整体替换成显式距离表**：`retrieve` 事件必须带 `distances`，长度与当前技能数一致；
  排序规则写成「距离升序、并列按插入序」，对应 Chroma 返回的有序结果。
- **落盘 IO 替换成 `dump_files` 集合**：只保留「V 号探测看的是目录里已有哪些文件」这一条语义，
  不真的写文件；返回的 `dumps[i]["file"]` 是不带 `.js` 的主名（官方也是先算主名再拼后缀）。
- **LLM 生成描述整体替换成入参**：`llm_description` 就是 `self.llm(messages).content`，
  模板包装仍在函数内做——这是本题要考的点，不外包给调用方。
- **`skills.json` 持久化 / `vectordb.persist()` 剔除**：属 IO，不影响账。
- **`retrieve_skills` 的返回类型**：官方只回 `codes`，本题额外记 `names` 与 `k`，因为「取了几条」
  和「按什么顺序」是这题的核心判据。
- **`synced` 保留**：它是官方那两条 `assert` 的镜像。因为覆盖分支先 delete 再 add，该字段在
  任何合法输入下恒为 `True`，仅作「向量库与字典同步」的对齐说明，**不构成判据**
  （见下方等价变异）。

## 变异

27 条，26 条 CAUGHT，1 条实证等价：

```
m-guard-short             前缀放宽成 "Deposit"             → check_guard_prefix 抓
m-guard-lower             前缀比对小写化                   → check_guard_prefix 抓
m-guard-fallthrough       跳过但不 continue                → check_guard_prefix 抓
m-skip-no-name            skipped 不记 program_name        → check_guard_prefix 抓
m-rewrite-false           覆盖当成新增                     → check_same_name_overwrites 抓
m-version-start-1         V 号从 1 起                      → check_version_suffix_scan 抓
m-version-no-scan         不做空位探测                     → check_version_suffix_scan 抓
m-dump-no-ext             登记文件名漏掉 .js               → check_version_suffix_scan 抓
m-new-skill-versioned     新技能也加 V 号                  → check_version_suffix_scan 抓
m-new-skill-no-order      新技能不进 order                 → 多条用例抓
m-book-raw-llm            描述不套模板                     → check_description_template 抓
m-desc-indent2            注释缩进改成两空格               → check_description_template 抓
m-desc-no-comment         注释不写 "// "                  → check_description_template 抓
m-k-no-min                k 不夹到库大小                  → check_retrieve_top_k 抓
m-tie-late-first          并列改成后到先得                 → check_retrieve_tie_by_insertion 抓
m-take-tail               取尾部 k 个而不是头部            → check_retrieve_top_k 抓
m-dist-len-off            不校验距离条数                   → check_validation 抓
m-dist-bool-ok            距离放过 bool                    → check_validation 抓
m-topk-bool-ok            top_k 放过 bool                  → check_validation 抓
m-topk-min-zero           top_k 允许 0                     → check_validation 抓
m-prims-first             原语拼在技能前面                 → check_programs_concatenation 抓
m-skill-sep-one           技能分隔符少一个换行             → check_programs_concatenation 抓
m-prim-sep-one            原语分隔符少一个换行             → check_programs_concatenation 抓
m-dup-name-ok             重名不报错                       → check_validation 抓
m-unknown-kind-ok         未知事件类型放过                 → check_validation 抓
m-count-dumps             count 数落盘次数                 → check_same_name_overwrites 抓
m-synced-true             synced 恒 True                  → 等价（见下）
```

**等价变异的实证**：`synced` 定义为 `len(book) == len(order)`，而 `book` 与 `order` 只在
「新技能」分支同时增长（L86-87 的 `dumped = name` 与 `order.append(name)` 同属一个 `else`），
覆盖分支两者都不动，起始状态两者都由 `skills` 同长度初始化，且 `skills` 内部重名会先被拒。
于是任何合法调用下 `len(book) == len(order)` 恒成立，常量 `True` 与原式不可区分——这不是漏测，
而是官方 L73 先 `delete(ids=[...])` 再 `add_texts(ids=[...])` 的设计后果（count 不涨）。

## 已知坑

- 第一版 10 条用例全写成 `def check_xxx():`（无参），直接 `TypeError: takes 0 positional
  arguments but 1 was given`；harness 与 selftest 都会传 `solve`，必须带参数。
- `check_against_oracle` 里 `dumps` 的期望顺序一开始写成 `["mineV2","craft","sword"]`，
  实际是 `["craft","mineV2","sword"]`——覆盖事件的落盘顺序由**事件顺序**决定，不是按名字重排。
- 「新技能也加 V 号」这条变异用 `craft` 的用例抓：`dump_files=["craft.js","craftV2.js"]`
  但库里没有 `craft`，第一次落盘必须是 `craft`。若只在「库里有同名」的场景测，这条变异会漏。
- `programs` 的最后一段也带 `"\n\n"`，非空结果永远以两个换行结尾；断言里别顺手 `rstrip`。

## 查重

- `ag-generative-memory-retrieval`：也是「按查询检索 top-k」，但对象是对话记忆、判据是
  相似度阈值与打分，没有同名覆盖、版本化落盘与提示词拼接。
- `ag-memorybank-forget-ledger`：记忆库的遗忘曲线与淘汰，条目只减不覆盖。
- `ag-ace-skillbook-ledger`(750)：同样叫「技能书」，但那是**策略文档按 delta 操作改写 + 渲染**，
  没有落盘命名与向量检索。
- `ag-funsearch-island-ledger`(748)、`ag-dgm-archive-ledger`(749)：程序库的进化池采样与存档选择，
  条目永不覆盖、也没有 top-k 检索。
- `peft-trainable-tokens-replace` 只是名字里有 token，机制无关。
