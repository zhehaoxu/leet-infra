"""Zep/Graphiti 的双时态边失效账。

把 Graphiti 一次「新事实入库」摊成可逐项核对的账本：去重（新建边还是并进
旧边）→ 打时间戳 → 判旧边该不该失效。

你实现的 `solve` 返回：

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
  "writes": [{"uuid", "field", "value"}],
}
```

三条时间轴的分工、以及两处容易写反的判据（严格大于 / 含等号）见
`solution.py` 的 docstring 与题面。
"""


def solve(extracted, related_edges=(), existing_edges=(), decisions=None,
          now=None, timestamps=None):
    raise NotImplementedError
