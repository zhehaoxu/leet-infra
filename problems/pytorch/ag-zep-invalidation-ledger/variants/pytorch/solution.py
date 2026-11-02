"""Zep/Graphiti：双时态边失效账（arXiv 2501.13956 §2.3）。

Graphiti 给每条事实边维护两对时间戳：

    valid_at / invalid_at    事件时间——事实在世界里何时成立、何时不再成立
    created_at / expired_at  事务时间——这条记录何时被写进图、何时被判定过期

新事实进来时先做去重（dedupe），再看它是否**顶掉**某条旧事实。顶掉的判据
全部落在事件时间上，逐条实现自 `graphiti_core/utils/maintenance/
edge_operations.py` 的 `_apply_temporal_invalidation` 与
`resolve_edge_contradictions`：

1. 新边自己带 `invalid_at`（说的是过去的事）且没 `expired_at`：立刻把
   `expired_at` 记成 now。
2. 否则，若新边还没过期，按 `(valid_at is None, valid_at)` 升序扫候选
   （**None 排最后**，稳定排序）：第一个 `valid_at` **严格大于**新边的候选
   会顶掉新边——新边的 `invalid_at` 取那个候选的 `valid_at`，`expired_at`
   记成 now，然后**停**。
3. 再逐条判候选该不该失效：
   - 候选早就失效、且失效时刻不晚于新边生效时刻 → 跳过（含等号）；
   - 新边在新边失效之前就失效了（新边更早）→ 跳过（含等号）；
   - 否则候选生效**严格早于**新边 → 候选 `invalid_at = 新边.valid_at`，
     `expired_at` 只在原本为空时写 now。
   - 平局（同一 `valid_at`）不失效。

第 2 步会**原地重排**候选列表，所以第 3 步的遍历顺序是排过序的。第 2 步
给新边写下的 `invalid_at` 又会喂进第 3 步的第二个跳过条件——这正是本题最
容易漏的耦合：新边被某个更晚的候选顶掉之后，其余更早的候选就都不再被它
失效了。

去重侧（同文件 `_dedupe_extracted_edge`）只决定「这次是新建边还是并进旧
边」，以及把 LLM 给的**连续下标**翻译回边：`0..len(related)-1` 指向
related，`len(related)..len(related)+len(existing)-1` 指向 existing。下标
越界不报错，只丢进告警账。`related` 与 `existing` 都为空时整段跳过（连
时间戳都不抽）；fact 归一化后逐字相同、且两端点相同则走快速路径，直接复用
旧边并**跳过**矛盾处理。
"""

import re
from datetime import datetime, timezone

_UTC = timezone.utc


def _parse(value):
    """ISO 串 -> aware UTC datetime；naive 按 UTC 解释（同 ensure_utc）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_UTC)


def _fmt(dt):
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _norm_fact(text):
    """同 dedup_helpers._normalize_string_exact：小写 + 空白折叠 + 剥边。"""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _row(state):
    return {
        "uuid": state["uuid"],
        "fact": state["fact"],
        "valid_at": _fmt(state["valid"]),
        "invalid_at": _fmt(state["invalid"]),
        "expired_at": _fmt(state["expired"]),
    }


def solve(extracted, related_edges=(), existing_edges=(), decisions=None,
          now=None, timestamps=None):
    """把一次「新边入库」摊成双时态账本。

    参数
    ----
    extracted : dict        新抽到的边 {"uuid","fact","source","target",
                            "valid_at","invalid_at"}（"expired_at" 视作 None）
    related_edges : list    同实体对的既有边（既做去重候选，也做失效候选）
    existing_edges : list   更广的失效候选（只做失效，不参与去重）
    decisions : dict        {"duplicate_facts": [int], "contradicted_facts": [int]}
                            两段下标连续编号，related 在前、existing 在后
    now : str               本次入库的事务时刻（utc_now 显式化，必填）
    timestamps : dict|None  {"valid_at","invalid_at"}：时间戳抽取 LLM 的返回，
                            只在 resolved 是新建边且原本两个时间戳都为空时生效

    返回
    ----
    {
      "path": "empty"|"fast"|"resolve",
      "resolved": {"uuid","fact","valid_at","invalid_at","expired_at"},
      "resolved_is_new": bool,
      "duplicate_ids": [int],              # 有效且保序，可重复
      "duplicate_uuids": [str],
      "invalid_duplicate_ids": [int],      # 越界的 duplicate 下标（原值）
      "invalid_contradiction_ids": [int],  # 越界的 contradicted 下标（原值）
      "candidate_uuids": [str],            # 解析后的失效候选，决策原序
      "sorted_candidate_uuids": [str],     # 排序后的遍历序（未排序时同原序）
      "invalidated": [{"uuid","valid_at","invalid_at","expired_at"}],
      "edges": {uuid: {"uuid","fact","valid_at","invalid_at","expired_at"}},
      "writes": [{"uuid","field","value"}],  # 每次写库事件，按发生顺序
    }
    """
    if now is None:
        raise ValueError("now")
    now_dt = _parse(now)

    related = list(related_edges)
    existing = list(existing_edges)

    state, order = {}, []

    def put(edge):
        uid = edge["uuid"]
        row = {
            "uuid": uid,
            "fact": edge["fact"],
            "valid": _parse(edge.get("valid_at")),
            "invalid": _parse(edge.get("invalid_at")),
            "expired": _parse(edge.get("expired_at")),
            "source": edge.get("source"),
            "target": edge.get("target"),
        }
        state[uid] = row
        order.append(uid)
        return row

    new_edge = put(extracted)
    for edge in related:
        put(edge)
    for edge in existing:
        put(edge)

    writes = []

    def write(row, field, value):
        row[field] = value
        writes.append({
            "uuid": row["uuid"],
            "field": field + "_at",
            "value": _fmt(value),
        })

    def apply_timestamps(row):
        # 官方 _extract_edge_timestamps 的门：已有任一时间戳就整段不抽。
        if row["valid"] is not None or row["invalid"] is not None:
            return
        if not timestamps:
            return
        if timestamps.get("valid_at"):
            write(row, "valid", _parse(timestamps["valid_at"]))
        if timestamps.get("invalid_at"):
            write(row, "invalid", _parse(timestamps["invalid_at"]))

    def package(path, resolved, dup_ids=(), invalid_dups=(),
                invalid_contras=(), cands=(), ordered=(), invalidated=()):
        return {
            "path": path,
            "resolved": _row(resolved),
            "resolved_is_new": resolved["uuid"] == new_edge["uuid"],
            "duplicate_ids": list(dup_ids),
            "duplicate_uuids": [related[i]["uuid"] for i in dup_ids],
            "invalid_duplicate_ids": list(invalid_dups),
            "invalid_contradiction_ids": list(invalid_contras),
            "candidate_uuids": [c["uuid"] for c in cands],
            "sorted_candidate_uuids": [c["uuid"] for c in ordered],
            "invalidated": [_row(c) for c in invalidated],
            "edges": {uid: _row(state[uid]) for uid in order},
            "writes": writes,
        }

    # ---- 无候选：不比对、不失效，但仍要抽时间戳 ------------------------
    if not related and not existing:
        apply_timestamps(new_edge)
        return package("empty", new_edge)

    # ---- 快速路径：fact 逐字相同且两端点相同，直接复用旧边 ---------------
    normalized = _norm_fact(extracted["fact"])
    for edge in related:
        if (edge.get("source") == extracted.get("source")
                and edge.get("target") == extracted.get("target")
                and _norm_fact(edge["fact"]) == normalized):
            return package("fast", state[edge["uuid"]])

    # ---- 常规路径：翻译 LLM 决策 ---------------------------------------
    if decisions is None:
        raise ValueError("decisions")

    dup_raw = list(decisions.get("duplicate_facts") or [])
    con_raw = list(decisions.get("contradicted_facts") or [])

    n_related = len(related)
    max_idx = n_related + len(existing) - 1

    invalid_dups = [i for i in dup_raw if i < 0 or i >= n_related]
    dup_ids = [i for i in dup_raw if 0 <= i < n_related]
    invalid_contras = [i for i in con_raw if i < 0 or i > max_idx]

    resolved = new_edge
    for idx in dup_ids:
        resolved = state[related[idx]["uuid"]]
        break

    cands = []
    for idx in con_raw:
        if 0 <= idx < n_related:
            cands.append(state[related[idx]["uuid"]])
        elif n_related <= idx <= max_idx:
            cands.append(state[existing[idx - n_related]["uuid"]])

    # 只给新建边抽时间戳；并进旧边时保留旧边的那一对。
    if resolved["uuid"] == new_edge["uuid"]:
        apply_timestamps(resolved)

    # ---- 第 1 步：新边自述已是历史 --------------------------------------
    if resolved["invalid"] is not None and resolved["expired"] is None:
        write(resolved, "expired", now_dt)

    # ---- 第 2 步：更晚的候选顶掉新边（原地排序，影响第 3 步遍历序）-------
    ordered = list(cands)
    if resolved["expired"] is None:
        ordered.sort(key=lambda c: (c["valid"] is None, c["valid"]))
        for cand in ordered:
            if (cand["valid"] is not None
                    and resolved["valid"] is not None
                    and cand["valid"] > resolved["valid"]):
                write(resolved, "invalid", cand["valid"])
                write(resolved, "expired", now_dt)
                break

    # ---- 第 3 步：逐条判候选该不该失效 ----------------------------------
    invalidated = []
    for cand in ordered:
        if (cand["invalid"] is not None
                and resolved["valid"] is not None
                and cand["invalid"] <= resolved["valid"]):
            continue
        if (cand["valid"] is not None
                and resolved["invalid"] is not None
                and resolved["invalid"] <= cand["valid"]):
            continue
        if (cand["valid"] is not None
                and resolved["valid"] is not None
                and cand["valid"] < resolved["valid"]):
            write(cand, "invalid", resolved["valid"])
            if cand["expired"] is None:
                write(cand, "expired", now_dt)
            invalidated.append(cand)

    return package("resolve", resolved, dup_ids, invalid_dups, invalid_contras,
                   cands, ordered, invalidated)
