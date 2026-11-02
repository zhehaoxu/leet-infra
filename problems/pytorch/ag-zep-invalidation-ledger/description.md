# Zep 双时态边失效账

Zep 的时序知识图（arXiv 2501.13956，实现即 `getzep/graphiti`）给每条事实边
维护**两对**时间戳：

- `valid_at` / `invalid_at` 是**事件时间**——这条事实在世界里何时成立、何时
  不再成立；
- `expired_at` 是**事务时间**——这条记录何时被系统判定「不再为真」。

两者不能混：一句「Alice 2024-03 搬到柏林」把「Alice 住在巴黎」顶掉时，旧边的
`invalid_at` 应该写 **2024-03**（事件发生的时间），而 `expired_at` 写**这次入库
的时刻**（事务时间）。

本题目把一次「新事实入库」剥成纯函数：LLM 的去重/矛盾判断变成 `decisions` 一条
入参，时间戳抽取变成 `timestamps` 一条入参，事务时刻变成 `now`，其余全是
**下标翻译、去重、失效判定**。实现

```
solve(extracted, related_edges=(), existing_edges=(), decisions=None,
      now=None, timestamps=None)
```

## 输入

- `extracted`：新抽到的边
  `{"uuid", "fact", "source", "target", "valid_at", "invalid_at"}`。
- `related_edges`：同实体对的既有边。既做**去重**候选，也做失效候选。
- `existing_edges`：更广的失效候选。**只**做失效，不参与去重。
- `decisions`：LLM 的返回
  `{"duplicate_facts": [int], "contradicted_facts": [int]}`。两段下标**连续编号**：
  `0..len(related_edges)-1` 指向 `related_edges`，
  `len(related_edges)..len(related_edges)+len(existing_edges)-1` 指向
  `existing_edges`。
- `now`：本次入库的事务时刻（必填，UTC ISO 串）。
- `timestamps`：时间戳抽取的返回 `{"valid_at", "invalid_at"}`，可缺省。

两边列表的 `uuid` 互不重复，且都与 `extracted` 不同。

## 输出

```
{
  "path": "empty" | "fast" | "resolve",
  "resolved": {"uuid", "fact", "valid_at", "invalid_at", "expired_at"},
  "resolved_is_new": bool,
  "duplicate_ids": [int],
  "duplicate_uuids": [str],
  "invalid_duplicate_ids": [int],
  "invalid_contradiction_ids": [int],
  "candidate_uuids": [str],
  "sorted_candidate_uuids": [str],
  "invalidated": [{"uuid", "valid_at", "invalid_at", "expired_at"}],
  "edges": {uuid: {"uuid", "fact", "valid_at", "invalid_at", "expired_at"}},
  "writes": [{"uuid", "field", "value"}]
}
```

`edges` 是**所有经手边**（新边 + 两个候选列表）在本次调用结束时的快照，
`invalidated` 是其中被判定失效的那几条（按遍历顺序）。所有时间字段输出统一为
UTC ISO 串（`...Z`），缺失一律是 `null`。

`writes` 是账本的审计流：每发生一次写库就追加一条，`field` 取
`"valid_at"` / `"invalid_at"` / `"expired_at"`，`value` 是写入后的值。

## 三条路径

1. **`empty`**：两个候选列表都空。没有可比的边——**连快速路径都不走**，直接
   给新边抽时间戳（见下），不做任何失效。
2. **`fast`**：`related_edges` 里存在一条边，其 `fact` 归一化后与新边**逐字相同**，
   且 `source`、`target` 都与新边相同。此时直接复用那条旧边：`resolved` 指向它、
   `duplicate_ids` 为空（不是 `[i]`）、完全不解析 `decisions`、不失效任何人。
   事实归一化 = 小写 → 连续空白折叠成单空格 → 剥首尾空白。
   注意快速路径**只扫 `related_edges`**。
3. **`resolve`**：其余情况。按下标翻译决策，再走去重与失效。

## 去重

- `duplicate_facts` 里越界（`< 0` 或 `>= len(related_edges)`）的下标不报错，
  原值记进 `invalid_duplicate_ids`；有效的保留**原顺序**，允许重复。
- `resolved` 取**第一个**有效下标指向的旧边；若一个都没有，`resolved` 就是新边。
- `duplicate_uuids` 与 `duplicate_ids` 一一对应，可以重复。
- **只有 `resolved` 是新边时**才应用 `timestamps`，并进旧边时旧边的时间戳原样保留。
- 应用 `timestamps` 前还有一道门：`resolved` 的 `valid_at`、`invalid_at` **只要
  有一个非空就整段跳过**。两个字段各自独立判断「非空串才写」，写入顺序固定
  `valid_at` 在前。

## 失效判定

`contradicted_facts` 里越界（`< 0` 或 `> len(related_edges) + len(existing_edges) - 1`）
的下标同样只记账，原值进 `invalid_contradiction_ids`。有效的按下标翻译成候选
（`related_edges` 段在前、`existing_edges` 段在后），**保留原顺序、允许重复**，
这就是 `candidate_uuids`。

然后三步：

1. **新边自述已是历史**：若 `resolved.invalid_at` 非空而 `expired_at` 为空，
   把 `expired_at` 记为 `now`。
2. **更晚的候选顶掉新边**：仅当 `resolved.expired_at` 仍为空时，把候选按
   `(valid_at 是否为空, valid_at)` **升序稳定排序**（`valid_at` 缺失的排**最后**），
   扫第一个 `valid_at` **严格大于** `resolved.valid_at` 的候选；命中就把
   `resolved.invalid_at` 写成它的 `valid_at`、`expired_at` 写成 `now`，然后**停**。
   这一步会**原地改写候选顺序**，第 3 步与 `sorted_candidate_uuids` 用的就是这个
   排过序的序列；若第 1 步已经落过 `expired_at`，则**不排序**，两者都保持原序。
3. **逐条判候选**。对排序后的每个候选（重复下标会重复进来，但下面第一条会把它
   挡掉）：
   - 候选自己已经失效、且失效时刻 `<= resolved.valid_at` → 跳过（**含等号**）；
   - 新边自己已经失效、且 `resolved.invalid_at <= 候选.valid_at` → 跳过（**含等号**，
     即新边比候选还早就不成立，没资格顶它）；
   - 否则，候选 `valid_at` **严格小于** `resolved.valid_at` 时失效它：
     `invalid_at` 写成 `resolved.valid_at`；`expired_at` **只在原本为空时**写 `now`，
     已有值保留。

平局（候选 `valid_at` 恰等于 `resolved.valid_at`）谁都不失效谁。

## 样例

```python
extracted = {"uuid": "new", "fact": "Alice lives in Berlin",
             "source": "alice", "target": "berlin",
             "valid_at": "2024-03-01T00:00:00Z", "invalid_at": None}
related_edges = [{"uuid": "old", "fact": "Alice lives in Paris",
                  "source": "alice", "target": "berlin",
                  "valid_at": "2024-01-01T00:00:00Z", "invalid_at": None,
                  "expired_at": None}]
decisions = {"duplicate_facts": [], "contradicted_facts": [0]}
now = "2024-06-01T00:00:00Z"

返回: path="resolve", resolved.valid_at="2024-03-01T00:00:00Z",
      invalidated=[{"uuid": "old", "valid_at": "2024-01-01T00:00:00Z",
                    "invalid_at": "2024-03-01T00:00:00Z",
                    "expired_at": "2024-06-01T00:00:00Z"}]
```

## 约束

- 纯函数：不联网、不读写文件、不修改入参。
- `now` 缺省、或在有候选边时 `decisions` 缺省，一律抛 `ValueError`。
- 时间入参是 ISO 串（`Z` 或 `+00:00` 均可）；不带时区的按 UTC 解释。
