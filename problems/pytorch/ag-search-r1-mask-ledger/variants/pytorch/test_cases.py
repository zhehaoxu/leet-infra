import random

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：动作判定逐位置扫描、截断用 list.index、
# 训练侧先用 (kind, id) 展平再切前 P 个、上下文侧另取后 P 个。

_PAIRS = {"<search>": ("search", "</search>"),
          "<answer>": ("answer", "</answer>")}


def _check_ids(ids, vocab):
    for x in ids:
        if x not in vocab:
            raise ValueError("unknown token id")


def _cut_one(ids, vocab):
    toks = [vocab[x] for x in ids]
    if "</search>" in toks:
        return ids[:toks.index("</search>") + 1]
    if "</answer>" in toks:
        return ids[:toks.index("</answer>") + 1]
    return ids


def _act_one(ids, vocab):
    toks = [vocab[x] for x in ids]
    i = 0
    while i < len(toks):
        if toks[i] in _PAIRS:
            kind, close = _PAIRS[toks[i]]
            j = i + 1
            while j < len(toks) and toks[j] != close:
                j += 1
            if j < len(toks):
                raw = "".join(vocab[x] for x in ids[i + 1:j])
                return kind, raw.strip()
        i += 1
    return None, ""


def _oracle(rows, vocab, search_results=None, wraps=None, max_turns=3,
            max_start_length=512, max_prompt_length=64, max_obs_length=32,
            drop_toks=("<|im_end|>", "<|endoftext|>")):
    if search_results is None:
        search_results = {}
    if wraps is None:
        wraps = {}
    for b in (max_turns, max_start_length, max_prompt_length, max_obs_length):
        if b < 0:
            raise ValueError("negative bound")
    drop = set(drop_toks)
    pre = list(wraps.get("obs_prefix", ()))
    suf = list(wraps.get("obs_suffix", ()))
    retry = list(wraps.get("retry", ()))

    n = len(rows)
    st = [{"used": 0, "acts": [], "cons": [], "kinds": [], "cuts": [],
           "sel": [], "turns": 1, "va": 0, "vs": 0, "live": True}
          for _ in range(n)]
    anl = [n]

    def _obs(tag, kind, ids):
        if max_obs_length < len(ids):
            kept = ids[:max_obs_length]
            tag["cuts"].append(len(ids) - max_obs_length)
        else:
            kept = list(ids)
            tag["cuts"].append(0)
        tag["kinds"].append(kind)
        if kept:
            tag["sel"].append(("obs", kept))

    step = 0
    while step < max_turns:
        if sum(1 for s in st if s["live"]) == 0:
            break
        step += 1
        for i in range(n):
            s = st[i]
            if not s["live"]:
                continue
            gens = rows[i].get("gens", [])
            if s["used"] >= len(gens):
                raise ValueError("missing generation")
            raw = list(gens[s["used"]])
            s["used"] += 1
            _check_ids(raw, vocab)
            keep = [x for x in raw if vocab[x] not in drop]
            pol = _cut_one(keep, vocab)
            kind, con = _act_one(pol, vocab)
            s["acts"].append(kind)
            s["cons"].append(con)
            s["sel"].append(("policy", pol))
            if kind == "answer":
                s["va"] += 1
                s["done"] = True
                s["kinds"].append("none")
                s["cuts"].append(0)
            elif kind == "search":
                if con not in search_results:
                    raise ValueError("no result")
                s["va"] += 1
                s["vs"] += 1
                _obs(s, "info", pre + list(search_results[con]) + suf)
            else:
                _obs(s, "retry", retry)
        for s in st:
            if s["live"] and not s.get("done"):
                s["turns"] += 1
        for s in st:
            if s.get("done"):
                s["live"] = False
                s["done"] = False
        anl.append(sum(1 for s in st if s["live"]))

    if sum(1 for s in st if s["live"]):
        for i in range(n):
            s = st[i]
            if not s["live"]:
                continue
            gens = rows[i].get("gens", [])
            if s["used"] >= len(gens):
                raise ValueError("missing generation")
            raw = list(gens[s["used"]])
            s["used"] += 1
            _check_ids(raw, vocab)
            keep = [x for x in raw if vocab[x] not in drop]
            pol = _cut_one(keep, vocab)
            kind, con = _act_one(pol, vocab)
            s["acts"].append(kind)
            s["cons"].append(con)
            s["kinds"].append("none")
            s["cuts"].append(0)
            s["sel"].append(("policy", pol))
            if kind == "answer":
                s["va"] += 1
                s["live"] = False
            elif kind == "search":
                s["va"] += 1
                s["vs"] += 1
        anl.append(sum(1 for s in st if s["live"]))

    out = []
    for i in range(n):
        s = st[i]
        prompt = list(rows[i].get("prompt", []))
        if max_start_length == 0 or max_start_length >= len(prompt):
            left = list(prompt)          # `[-0:]` 与越界负下标都是全量
        else:
            left = prompt[len(prompt) - max_start_length:]
        flat = []
        for kind, ids in s["sel"]:
            for x in ids:
                flat.append((kind, x))
        kept = flat[:max_prompt_length]
        dropped = {"policy": 0, "obs": 0}
        for kind, _x in flat[max_prompt_length:]:
            dropped[kind] += 1
        segs = []
        for kind, _x in kept:
            if segs and segs[-1][0] == kind:
                segs[-1][1] += 1
            else:
                segs.append([kind, 1])
        produced = [x for _k, x in flat]
        if max_prompt_length == 0:
            roll = prompt + produced
        else:
            roll = (prompt + produced)[-max_prompt_length:]
        out.append({
            "actions": s["acts"],
            "contents": s["cons"],
            "obs_kinds": s["kinds"],
            "obs_truncated": s["cuts"],
            "turns": s["turns"],
            "valid_actions": s["va"],
            "valid_searches": s["vs"],
            "finished": not s["live"],
            "right_ids": [x for _k, x in kept],
            "segments": segs,
            "dropped": dropped,
            "final_ids": left + [x for _k, x in kept],
            "final_info_mask": [1] * len(left)
            + [1 if k == "policy" else 0 for k, _x in kept],
            "rolling_ids": roll,
        })
    return {"rows": out, "active_num_list": anl}


# ---------------------------------------------------------------- compare

def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            _eq(x, y)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (a.keys() ^ b.keys())
        for k in a:
            _eq(a[k], b[k])
        return
    assert a == b, (a, b)


def _compare(solve, *args, **kw):
    got = solve(*[_dc(x) for x in args], **deepcopy(kw))
    exp = _oracle(*[_dc(x) for x in args], **deepcopy(kw))
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- fixtures
# 词表：1..4 是四个标签、5/6 是两个「特殊 token」、7..11 是观测包装与
# 重试提示、20 起是普通词。
_V = {1: "<search>", 2: "</search>", 3: "<answer>", 4: "</answer>",
      5: "<|im_end|>", 6: "<|endoftext|>",
      7: "\n\n", 8: "<information>", 9: "</information>\n\n",
      10: "<retry>", 11: "RETRY ",
      20: "Q", 21: " P", 30: "weather", 31: "42", 32: "paris",
      40: "Doc1", 41: " Doc2", 50: " then ", 51: "a", 52: " b "}
_W = {"obs_prefix": [7, 8], "obs_suffix": [9], "retry": [10, 11]}
_SR = {"weather": [40, 41], "paris": [40], "weather<search>42": [40],
       "b  b": [41], "42": [40]}
_SR_LONG = dict(_SR, weather=list(range(60, 80)))


def _led(solve, rows, sr=None, w=None, v=None, **kw):
    return _compare(solve, rows, v if v is not None else _V,
                    _SR if sr is None else sr,
                    _W if w is None else w, **kw)


def check_search_priority_cut(solve):
    """截断按 `</search>` **优先且与位置无关**：先答后搜时答案是动作，
    但 search 标签整段留在 policy token 里（官方没把它当结束符）。"""
    led = _led(solve, [{"prompt": [20], "gens": [[3, 31, 4, 50, 1, 30, 2]]}],
               max_turns=2)
    r = led["rows"][0]
    assert r["actions"] == ["answer"]
    assert r["contents"] == ["42"]
    assert r["obs_kinds"] == ["none"]
    # 7 个 token 全留：`</search>` 优先于更靠前的 `</answer>`
    assert r["right_ids"] == [3, 31, 4, 50, 1, 30, 2]
    assert r["segments"] == [["policy", 7]]
    # 反过来（先搜后答）：`</search>` 截断把后面的 answer 全砍掉
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2, 3, 31, 4], [3, 31, 4]]}],
               max_turns=2)
    r = led["rows"][0]
    assert r["actions"] == ["search", "answer"]
    assert r["right_ids"][:3] == [1, 30, 2]
    assert r["segments"] == [["policy", 3], ["obs", 5], ["policy", 3]]


def check_parse_matched_pair(solve):
    """解析要**配对**（`<(search|answer)>…</\\1>`）：只有闭标签、开标签
    没闭合都不算动作；同名嵌套取最近闭标签；内容 strip。"""
    # `<answer>42 then</search>`：有 `</search>` 故截断不砍，但没有
    # `</answer>` → 动作 None → 走重试
    led = _led(solve, [{"prompt": [20], "gens": [[3, 31, 50, 2], [3, 31, 4]]}],
               max_turns=2)
    r = led["rows"][0]
    assert r["actions"] == [None, "answer"]
    assert r["contents"] == ["", "42"]
    assert r["obs_kinds"] == ["retry", "none"]
    assert r["valid_actions"] == 1
    # 同名嵌套：最近闭标签决定内容
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 1, 31, 2], [3, 31, 4]]}],
               max_turns=2)
    r = led["rows"][0]
    assert r["contents"] == ["weather<search>42", "42"]
    # 内容 strip（两侧空格）
    led = _led(solve, [{"prompt": [20], "gens": [[3, 52, 4]]}], max_turns=1)
    assert led["rows"][0]["contents"] == ["b"]
    # 空内容也是合法 answer（不校验非空）
    led = _led(solve, [{"prompt": [20], "gens": [[3, 4]]}], max_turns=1)
    r = led["rows"][0]
    assert r["actions"] == ["answer"]
    assert r["finished"] is True


def check_invalid_retry(solve):
    """非法动作照样吃掉一轮、把重试提示当观测拼进去（掩码 0），
    valid_action 不加；那一轮的 policy token 仍参与 loss。"""
    led = _led(solve, [{"prompt": [20], "gens": [[21], [3, 31, 4]]}], max_turns=3)
    r = led["rows"][0]
    assert r["actions"] == [None, "answer"]
    assert r["obs_kinds"] == ["retry", "none"]
    assert r["turns"] == 2
    assert r["valid_actions"] == 1
    assert r["segments"] == [["policy", 1], ["obs", 2], ["policy", 3]]
    assert r["final_info_mask"] == [1, 1, 0, 0, 1, 1, 1]


def check_obs_cap_prefix(solve):
    """观测按 `[:, :max_obs_length]` **截前缀**：`</information>` 被砍掉，
    重试提示同样受这个上限约束。"""
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}],
               max_turns=2, max_obs_length=3)
    r = led["rows"][0]
    assert r["obs_truncated"] == [2, 0]
    assert r["segments"] == [["policy", 3], ["obs", 3], ["policy", 3]]
    assert r["right_ids"][3:6] == [7, 8, 40]
    led = _led(solve, [{"prompt": [20], "gens": [[21], [3, 31, 4]]}],
               max_turns=2, max_obs_length=1)
    r = led["rows"][0]
    assert r["obs_truncated"] == [1, 0]
    assert r["segments"] == [["policy", 1], ["obs", 1], ["policy", 3]]
    assert r["right_ids"][1] == 10


def check_final_rollout(solve):
    """主循环用尽后还有一次 `do_search=False` 的 final rollout：它不追加
    观测、不算 turns、search 仍记 valid_action 但永远不算完成。"""
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2], [1, 30, 2]]}],
               max_turns=1)
    r = led["rows"][0]
    assert r["actions"] == ["search", "search"]
    assert r["obs_kinds"] == ["info", "none"]
    assert r["turns"] == 2
    assert r["valid_searches"] == 2
    assert r["finished"] is False
    assert led["active_num_list"] == [1, 1, 1]
    # 第二次 search 只有 policy、没有观测段
    assert r["segments"] == [["policy", 3], ["obs", 5], ["policy", 3]]
    # final rollout 里答完可以完成，且同样不计 turns
    led = _led(solve, [{"prompt": [20], "gens": [[21], [3, 31, 4]]}], max_turns=1)
    r = led["rows"][0]
    assert r["turns"] == 2
    assert r["finished"] is True
    assert r["valid_actions"] == 1


def check_right_cap_prefix(solve):
    """训练侧 `_update_right_side` 是**前缀锚定**：攒满 max_prompt_length
    后新 token 静默丢弃（动作照常生效、照常完成），账上记 dropped。"""
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}],
               max_turns=2, max_prompt_length=5)
    r = led["rows"][0]
    assert r["actions"] == ["search", "answer"]
    assert r["finished"] is True
    assert r["right_ids"] == [1, 30, 2, 7, 8]
    assert r["segments"] == [["policy", 3], ["obs", 2]]
    assert r["dropped"] == {"policy": 3, "obs": 3}
    assert r["final_ids"] == [20, 1, 30, 2, 7, 8]
    assert len(r["final_info_mask"]) == 6


def check_two_windows(solve):
    """训练侧取**最老**的 P 个、上下文侧取**最新**的 P 个：同一个 rollout
    两个窗口内容不同（答案进了上下文、没进 loss）。"""
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}],
               max_turns=2, max_prompt_length=6)
    r = led["rows"][0]
    assert r["right_ids"] == [1, 30, 2, 7, 8, 40]
    # 上下文窗口 = prompt + 全部产出 的最后 6 个
    assert r["rolling_ids"] == [40, 41, 9, 3, 31, 4]
    assert r["rolling_ids"] != r["right_ids"]
    assert r["final_info_mask"][-1] == 0        # 训练侧连观测都没走完


def check_prompt_tail(solve):
    """左端 `initial_input_ids[:, -max_start_length:]` 是**后缀**截断；
    `max_start_length=0` 因 `[-0:] == [0:]` 反而保留整段 prompt。"""
    led = _led(solve, [{"prompt": [20, 21, 51], "gens": [[3, 31, 4]]}],
               max_turns=1, max_start_length=1)
    assert led["rows"][0]["final_ids"][:2] == [51, 3]
    led = _led(solve, [{"prompt": [20, 21, 51], "gens": [[3, 31, 4]]}],
               max_turns=1, max_start_length=0)
    assert led["rows"][0]["final_ids"][:4] == [20, 21, 51, 3]
    led = _led(solve, [{"prompt": [20], "gens": [[3, 31, 4]]}],
               max_turns=1, max_start_length=512)
    assert led["rows"][0]["final_ids"] == [20, 3, 31, 4]
    assert led["rows"][0]["final_info_mask"] == [1, 1, 1, 1]


def check_zero_prompt_length(solve):
    """`max_prompt_length=0` 同时是「右端 `[:0]` 空」与「上下文 `[-0:]`
    全量」——同一个 0 在两侧语义相反。"""
    led = _led(solve, [{"prompt": [20], "gens": [[3, 31, 4]]}],
               max_turns=1, max_prompt_length=0)
    r = led["rows"][0]
    assert r["right_ids"] == []
    assert r["segments"] == []
    assert r["dropped"] == {"policy": 3, "obs": 0}
    assert r["final_ids"] == [20]
    assert r["final_info_mask"] == [1]
    assert r["rolling_ids"] == [20, 3, 31, 4]


def check_decode_drop(solve):
    """`batch_decode(skip_special_tokens=True)` + 重新 tokenize：被丢的特殊
    token 从 policy 段整段消失（也就少了一个 loss 位）。"""
    led = _led(solve, [{"prompt": [20], "gens": [[3, 5, 31, 6, 4]]}], max_turns=1)
    r = led["rows"][0]
    assert r["right_ids"] == [3, 31, 4]
    assert r["segments"] == [["policy", 3]]
    assert r["contents"] == ["42"]
    # 丢掉之后标签才相邻：截断与解析都按丢弃后的序列走
    led = _led(solve, [{"prompt": [20], "gens": [[1, 30, 2, 5, 3, 31, 4],
                                                [3, 31, 4]]}], max_turns=1)
    r = led["rows"][0]
    assert r["actions"] == ["search", "answer"]
    assert r["right_ids"][:3] == [1, 30, 2]
    assert r["segments"][0] == ["policy", 3]
    assert 5 not in r["right_ids"]


def check_active_num_list(solve):
    """`active_num_list` = 初始活跃数 + 主循环每步之后 + （若还有活跃行）
    final rollout 之后；全员完成时**提前 break**，不多出一个尾项。"""
    rows = [{"prompt": [20], "gens": [[3, 31, 4], [3, 31, 4]]}]
    led = _led(solve, rows, max_turns=3)
    assert led["active_num_list"] == [1, 0]        # 一步完成即 break
    rows = [{"prompt": [20], "gens": [[1, 30, 2], [1, 30, 2], [3, 31, 4]]}]
    led = _led(solve, rows, max_turns=2)
    assert led["active_num_list"] == [1, 1, 1, 0]  # 2 步 + final
    rows = [{"prompt": [20], "gens": [[3, 31, 4]]},
            {"prompt": [20], "gens": [[1, 30, 2], [1, 30, 2], [1, 30, 2]]},
            {"prompt": [20], "gens": [[21], [21], [21]]}]
    led = _led(solve, rows, max_turns=2)
    assert led["active_num_list"] == [3, 2, 2, 2]
    assert [r["turns"] for r in led["rows"]] == [1, 3, 3]
    assert [r["finished"] for r in led["rows"]] == [True, False, False]


def check_errors(solve):
    """域约束：id 不在词表、某轮缺生成、query 没结果、边界为负。

    这里**直接调 solve**（不过 oracle）——否则参考解自己抛的错会替被测
    实现兜底，边界检查写成空操作也发现不了。
    """
    def _bad(fn):
        try:
            fn()
        except ValueError:
            return True
        except Exception as exc:                       # noqa: BLE001
            raise AssertionError("wrong error: %r" % (exc,))
        raise AssertionError("expected ValueError")

    def _call(rows, **kw):
        return solve(_dc(rows), _V, _SR, _W, **kw)

    assert _bad(lambda: _call([{"prompt": [20], "gens": [[3, 999, 4]]}],
                              max_turns=1))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[1, 30, 2]]}],
                              max_turns=3))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[1, 51, 2]]}],
                              max_turns=1))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[3, 31, 4]]}],
                              max_turns=-1))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[3, 31, 4]]}],
                              max_turns=1, max_prompt_length=-2))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[3, 31, 4]]}],
                              max_turns=1, max_obs_length=-1))
    assert _bad(lambda: _call([{"prompt": [20], "gens": [[3, 31, 4]]}],
                              max_turns=1, max_start_length=-5))
    # 缺生成只在真被消费时才算错：早早答完的行不需要多余生成
    led = _led(solve, [{"prompt": [20], "gens": [[3, 31, 4]]}], max_turns=5)
    assert led["rows"][0]["finished"] is True
    # 空 prompt 合法（右端从零开始攒）
    led = _led(solve, [{"prompt": [], "gens": [[3, 31, 4]]}], max_turns=1)
    assert led["rows"][0]["final_ids"] == [3, 31, 4]


def check_default_args(solve):
    """默认值必须与官方 CLI 一致：max_turns=3、max_start_length=512、
    max_prompt_length=64、max_obs_length=32、只丢两个特殊 token。"""
    # 一直搜 + 一次 final → turns = 3 + 1
    rows = [{"prompt": [20], "gens": [[1, 30, 2]] * 4}]
    led = _compare(solve, rows, _V, _SR, _W)
    r = led["rows"][0]
    assert r["turns"] == 4
    assert r["valid_searches"] == 4
    assert led["active_num_list"] == [1, 1, 1, 1, 1]
    # 默认 obs 上限 32：23 个 token 的观测不被砍
    led = _compare(solve, [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}],
                   _V, _SR_LONG, _W)
    r = led["rows"][0]
    assert r["obs_truncated"] == [0, 0]
    assert r["segments"] == [["policy", 3], ["obs", 23], ["policy", 3]]
    # 默认 prompt 上限 64：42 个生成 token 不封顶
    led = _compare(solve, [{"prompt": [20], "gens": [[3] + [31] * 40 + [4]]}],
                   _V, _SR, _W)
    assert len(led["rows"][0]["right_ids"]) == 42
    assert led["rows"][0]["dropped"] == {"policy": 0, "obs": 0}
    # 默认起点 512：300 token 的长 prompt 全留
    long_prompt = list(range(1000, 1300))
    led = _compare(solve, [{"prompt": long_prompt, "gens": [[3, 31, 4]]}],
                   _V, _SR, _W)
    assert len(led["rows"][0]["final_ids"]) == 303
    assert led["rows"][0]["final_ids"][0] == 1000


def check_random_cross(solve):
    """随机交叉：随机多行 rollout（含未闭合标签、特殊 token、非法动作）
    与随机构型，全账对齐。"""
    class _AnyDict(dict):
        def __contains__(self, k):
            return True

        def __getitem__(self, k):
            return dict.__getitem__(self, k) if dict.__contains__(self, k) \
                else [40]

    pool = [1, 2, 3, 4, 31, 32, 50, 51, 52, 5]
    sources = {"weather": [40, 41], "paris": [40]}
    for seed in range(60):
        rng = random.Random(seed)
        n_row = rng.randint(1, 3)
        turns = rng.randint(0, 3)
        rows = []
        for _ in range(n_row):
            gens = []
            for _g in range(turns + 1):
                ln = rng.randint(0, 7)
                gens.append([rng.choice(pool) for _ in range(ln)])
            rows.append({"prompt": [rng.choice([20, 21, 51])
                                    for _ in range(rng.randint(0, 3))],
                         "gens": gens})
        kw = dict(max_turns=turns,
                  max_start_length=rng.choice([0, 1, 2, 512]),
                  max_prompt_length=rng.choice([0, 3, 8, 64]),
                  max_obs_length=rng.choice([0, 1, 4, 32]))
        _compare(solve, rows, _V, _AnyDict(sources), _W, **kw)


def check_example(solve):
    """run 模式样例：一次检索 → 一次作答，检索段整体掩 0。"""
    rows = [{"prompt": [20, 21], "gens": [[1, 30, 2], [3, 31, 4]]}]
    led = _led(solve, rows, max_turns=3)
    r = led["rows"][0]
    assert r["actions"] == ["search", "answer"]
    assert r["contents"] == ["weather", "42"]
    assert r["turns"] == 2
    assert r["finished"] is True
    assert r["segments"] == [["policy", 3], ["obs", 5], ["policy", 3]]
    assert r["final_info_mask"] == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    assert led["active_num_list"] == [1, 1, 0]


check_example.example = True
