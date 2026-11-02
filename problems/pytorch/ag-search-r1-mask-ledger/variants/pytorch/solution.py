"""Search-R1 检索掩码账。

Search-R1（arXiv 2503.09516）的多轮 rollout 把「生成 → `<search>` →
观测」交错拼成一条序列，训练时要用 `info_mask` 把观测段剔出 loss。
回源 PeterGriffinJin/Search-R1 main：
`search_r1/llm_agent/generation.py`（`_postprocess_responses`、
`postprocess_predictions`、`execute_predictions`、`run_llm_loop`、
`_update_right_side`、`_update_rolling_state`、`_compose_final_output`）
与 `search_r1/llm_agent/tensor_helper.py`。

给定一批行的 prompt、逐次生成 token、检索结果与 harness 的观测包装
token，复算：每次生成的截断与动作解析、观测段的掩码、训练侧（前缀
锚定）与上下文侧（后缀锚定）两个窗口、以及 rollout 统计。

账目只记真实 token：左端 prompt pad 与右端补偿 pad 在 attention_mask
与 info_mask 中恒 0，不出现在账里（见 notes）。
"""

_SEARCH_OPEN = "<search>"
_SEARCH_CLOSE = "</search>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"


def _ids_to_toks(ids, vocab):
    out = []
    for i in ids:
        try:
            out.append(vocab[i])
        except KeyError:
            raise ValueError("unknown token id: %r" % (i,))
    return out


def _postprocess(ids, vocab, drop):
    """官方 `_postprocess_responses`：batch_decode(skip_special_tokens=True)
    后按第一个 `</search>` 截断、否则按第一个 `</answer>` 截断，再把闭标签
    补回；最后重新 tokenize（故被 drop 的特殊 token 整段消失）。

    search 分支**与位置无关地优先**：先看全文有没有 `</search>`。
    """
    toks = _ids_to_toks(ids, vocab)
    kept, ktoks = [], []
    for k in range(len(ids)):
        if toks[k] in drop:
            continue
        kept.append(ids[k])
        ktoks.append(toks[k])
    for tag in (_SEARCH_CLOSE, _ANSWER_CLOSE):
        if tag in ktoks:
            cut = ktoks.index(tag) + 1
            return kept[:cut]
    return kept


def _parse(ids, vocab):
    """官方 `postprocess_predictions`：`<(search|answer)>(.*?)</\\1>`
    （DOTALL、`re.search`）。带闭标签的**第一个**开标签胜出，内容取
    其后**最近**的同名闭标签之间的原文再 strip。"""
    toks = _ids_to_toks(ids, vocab)
    n = len(toks)
    for i in range(n):
        if toks[i] == _SEARCH_OPEN:
            kind, close = "search", _SEARCH_CLOSE
        elif toks[i] == _ANSWER_OPEN:
            kind, close = "answer", _ANSWER_CLOSE
        else:
            continue
        for j in range(i + 1, n):
            if toks[j] == close:
                inner = "".join(vocab[k] for k in ids[i + 1:j])
                return kind, inner.strip()
    return None, ""


def _clip(ids, limit):
    """`tensor[:, :limit]`：前缀锚定（保留开头），返回 (保留, 被砍个数)。"""
    if limit < 0 or limit >= len(ids):
        return list(ids), 0
    return list(ids[:limit]), len(ids) - limit


def solve(rows, vocab, search_results=None, wraps=None, max_turns=3,
          max_start_length=512, max_prompt_length=64, max_obs_length=32,
          drop_toks=("<|im_end|>", "<|endoftext|>")):
    if search_results is None:
        search_results = {}
    if wraps is None:
        wraps = {}
    obs_pre = list(wraps.get("obs_prefix", ()))
    obs_suf = list(wraps.get("obs_suffix", ()))
    retry = list(wraps.get("retry", ()))
    if max_turns < 0 or max_obs_length < 0 or max_prompt_length < 0 \
            or max_start_length < 0:
        raise ValueError("negative bound")
    drop = set(drop_toks)

    n = len(rows)
    cursor = [0] * n
    active = [True] * n
    turns = [1] * n
    n_valid_action = [0] * n
    n_valid_search = [0] * n
    actions = [[] for _ in range(n)]
    contents = [[] for _ in range(n)]
    obs_kinds = [[] for _ in range(n)]
    obs_cut = [[] for _ in range(n)]
    # 训练侧右端按 (kind, ids) 顺序攒；上下文侧只是同一串的另一个窗口
    right = [[] for _ in range(n)]

    def _emit(i, kind, ids):
        if ids:
            right[i].append((kind, list(ids)))

    def _obs(i, kind, ids):
        kept, cut = _clip(ids, max_obs_length)
        obs_kinds[i].append(kind)
        obs_cut[i].append(cut)
        _emit(i, "obs", kept)

    active_num_list = [sum(1 for a in active if a)]
    for _ in range(max_turns):
        if not any(active):
            break
        done_this = [False] * n
        for i in range(n):
            if not active[i]:
                continue
            gens = rows[i].get("gens", [])
            if cursor[i] >= len(gens):
                raise ValueError("row %d: missing generation #%d" % (i, cursor[i]))
            proc = _postprocess(list(gens[cursor[i]]), vocab, drop)
            cursor[i] += 1
            action, content = _parse(proc, vocab)
            actions[i].append(action)
            contents[i].append(content)
            _emit(i, "policy", proc)
            if action == "answer":
                n_valid_action[i] += 1
                done_this[i] = True
                obs_kinds[i].append("none")
                obs_cut[i].append(0)
            elif action == "search":
                if content not in search_results:
                    raise ValueError("no result for query: %r" % (content,))
                n_valid_action[i] += 1
                n_valid_search[i] += 1
                _obs(i, "info", obs_pre + list(search_results[content]) + obs_suf)
            else:
                _obs(i, "retry", retry)
        for i in range(n):
            if active[i] and not done_this[i]:
                turns[i] += 1
        for i in range(n):
            if done_this[i]:
                active[i] = False
        active_num_list.append(sum(1 for a in active if a))

    # final LLM rollout：`do_search=False`，next_obs 整个丢弃、不追加观测，
    # 也不更新 turns_stats；search 仍算 valid_action / is_search，但 done=0。
    if any(active):
        done_this = [False] * n
        for i in range(n):
            if not active[i]:
                continue
            gens = rows[i].get("gens", [])
            if cursor[i] >= len(gens):
                raise ValueError("row %d: missing generation #%d" % (i, cursor[i]))
            proc = _postprocess(list(gens[cursor[i]]), vocab, drop)
            cursor[i] += 1
            action, content = _parse(proc, vocab)
            actions[i].append(action)
            contents[i].append(content)
            obs_kinds[i].append("none")
            obs_cut[i].append(0)
            _emit(i, "policy", proc)
            if action == "answer":
                n_valid_action[i] += 1
                done_this[i] = True
            elif action == "search":
                n_valid_action[i] += 1
                n_valid_search[i] += 1
        for i in range(n):
            if done_this[i]:
                active[i] = False
        active_num_list.append(sum(1 for a in active if a))

    out = []
    for i in range(n):
        prompt = list(rows[i].get("prompt", []))
        left = prompt[-max_start_length:]
        merged = []  # 相邻同类段合并（final rollout 的 policy 紧跟上一次的 policy）
        for kind, ids in right[i]:
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + list(ids))
            else:
                merged.append((kind, list(ids)))
        kept, dropped = [], {"policy": 0, "obs": 0}
        budget = max_prompt_length
        for kind, ids in merged:
            take = min(budget, len(ids))
            if take:
                kept.append((kind, ids[:take]))
            dropped[kind] += len(ids) - take
            budget -= take
        right_ids = [x for _k, ids in kept for x in ids]
        segments = [[k, len(ids)] for k, ids in kept]
        mask = [1] * len(left)
        for k, ids in kept:
            mask.extend([1 if k == "policy" else 0] * len(ids))
        produced = [x for _k, ids in merged for x in ids]
        if max_prompt_length:
            roll_ids = (prompt + produced)[-max_prompt_length:]
        else:
            roll_ids = prompt + produced  # `[-0:]` 等于全量：0 不是零宽窗口
        out.append({
            "actions": actions[i],
            "contents": contents[i],
            "obs_kinds": obs_kinds[i],
            "obs_truncated": obs_cut[i],
            "turns": turns[i],
            "valid_actions": n_valid_action[i],
            "valid_searches": n_valid_search[i],
            "finished": not active[i],
            "right_ids": right_ids,
            "segments": segments,
            "dropped": dropped,
            "final_ids": left + right_ids,
            "final_info_mask": mask,
            "rolling_ids": roll_ids,
        })
    return {"rows": out, "active_num_list": active_num_list}
