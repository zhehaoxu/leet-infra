"""
Self-RAG 反思 token 打分账：把 Self-RAG（arXiv 2310.11511）推理期的
「反思 token 打分 + 证据重排聚合」无状态化。

回源：AkariAsai/self-rag main
- retrieval_lm/run_short_form.py::call_model_rerank_w_scores_batch
  （检索触发门、relevance/ground/utility 打分、final_score、
   answer2score / best_path 聚合）
- retrieval_lm/run_short_form.py::postprocess_answer_option_conditioned
- retrieval_lm/utils.py（reflection token 名单、postprocess）

solve(no_retr, evidences, second_text, mode="adaptive_retrieval",
      threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5,
      use_seqscore=False, use_groundness=False, use_utility=False,
      closed=False)

- no_retr：无检索那次生成 {"text": str, "lp0": {token: logprob}}
  （lp0 = **第 0 个生成位置**的 logprob 表）。
- evidences：[{"text", "tokens", "lps", "cumulative_logprob"}]，
  每条证据一段检索增强生成；tokens 是生成出的 token 串序列、
  lps[i] 是第 i 个位置的 logprob 表（长度须与 tokens 一致）。
- second_text：do_retrieve 为 False 时**第二次**生成（prompt 追加
  "[No Retrieval]"）的文本。

返回账：
{"do_retrieve": bool,
 "retrieval_ratio": float|None,   # lp[Retrieval]/(lp[Retrieval]+lp[No Retrieval])
 "result_keys": [str ...],        # results 的键序
 "evidence_scores": [{"idx", "relevance", "ground", "utility",
                      "seq_exp", "final"}, ...],
 "answer": str,
 "best_path": str|None,           # closed=False 时的最优路径键
 "answer2score": {answer: score}|None,   # closed=True
 "answer_score": float|None,
 "n_evidence": int}

域约束（ValueError）：threshold 路径下 lp0 缺反思 token（官方先赋
-100 死代码、随即 KeyError）；tokens 与 lps 长度不一致；do_retrieve
为 True 却没有证据（官方 UnboundLocalError / 聚合空列表 IndexError）。
"""
import math


RET_TOKENS = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
REL_TOKENS = ["[Irrelevant]", "[Relevant]"]
GRD_TOKENS = ["[Fully supported]", "[Partially supported]",
              "[No support / Contradictory]"]
UT_TOKENS = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]",
             "[Utility:5]"]
CONTROL_TOKENS = ["[Fully supported]", "[Partially supported]",
                  "[No support / Contradictory]", "[No Retrieval]",
                  "[Retrieval]", "[Irrelevant]", "[Relevant]",
                  "<paragraph>", "</paragraph>", "[Utility:1]",
                  "[Utility:2]", "[Utility:3]", "[Utility:4]",
                  "[Utility:5]"]
UT_WEIGHTS = [-1, -0.5, 0, 0.5, 1]
_MISSING_LOGPROB = -100.0     # 缺失 token 的兜底 logprob（官方 -100）

def _p(lp_table, token):
    """缺失 token 的 logprob 兜底 -100（官方 `if id in ... else -100`），
    再 exp 成概率。"""
    return math.exp(float(lp_table.get(token, _MISSING_LOGPROB)))


def _gate_ratio(lp0, threshold):
    score = {}
    for tok in RET_TOKENS:
        if tok not in lp0:
            # 官方此处先写兜底 -100、下一行再无条件下标取值：兜底是死
            # 代码，缺键当场 KeyError（本题收敛成 ValueError）
            raise ValueError("missing gate token: %s" % tok)
        score[tok] = float(lp0[tok])
    ratio = score["[Retrieval]"] / (score["[Retrieval]"]
                                    + score["[No Retrieval]"])
    return ratio, ratio > threshold


def _evidence_score(ev, idx, w_rel, w_sup, w_use, use_seqscore,
                    use_groundness, use_utility):
    lps0 = ev["lps"][0]
    rel = {}
    for tok in REL_TOKENS:
        rel[tok] = _p(lps0, tok)
    relevance = rel["[Relevant]"] / sum(rel.values())

    ground = 0.0
    if use_groundness:
        pos = None
        for k, tok in enumerate(ev["tokens"]):
            if tok in GRD_TOKENS:     # 官方遇到第一个即 break
                pos = k
                break
        if pos is not None:
            grd = {}
            for tok in GRD_TOKENS:
                grd[tok] = _p(ev["lps"][pos], tok)
            total = sum(grd.values())
            ground = (grd["[Fully supported]"] / total
                      + 0.5 * (grd["[Partially supported]"] / total))

    utility = 0.0
    if use_utility:
        pos = None
        for k, tok in enumerate(ev["tokens"]):
            if tok in UT_TOKENS:      # 官方此处**不** break（收全部再取首个）
                pos = k
                break
        if pos is not None:
            ut = {}
            for tok in UT_TOKENS:
                ut[tok] = _p(ev["lps"][pos], tok)
            total = sum(ut.values())
            utility = sum(UT_WEIGHTS[i] * (ut["[Utility:%d]" % (i + 1)]
                                           / total)
                          for i in range(len(UT_WEIGHTS)))

    seq_exp = math.exp(float(ev["cumulative_logprob"])
                       / max(len(ev["tokens"]), 1))
    final = w_rel * relevance + w_sup * ground + w_use * utility
    if use_seqscore:
        final += seq_exp
    return {"idx": idx, "relevance": relevance, "ground": ground,
            "utility": utility, "seq_exp": seq_exp, "final": final}


def _postprocess(text):
    for tok in CONTROL_TOKENS:
        text = text.replace(tok, "")
    text = text.replace("</s>", "")
    if len(text) == 0:
        return ""
    if text[0] == " ":
        text = text[1:]
    return text


def _postprocess_option(text):
    for tok in CONTROL_TOKENS:
        text = text.replace(tok, "")
    text = text.replace("</s>", "")
    text = text.replace("\n", "")
    text = text.replace("<|endoftext|>", "")
    return text


def solve(no_retr, evidences, second_text, mode="adaptive_retrieval",
          threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5,
          use_seqscore=False, use_groundness=False, use_utility=False,
          closed=False):
    for ev in evidences:
        if len(ev["tokens"]) != len(ev["lps"]):
            raise ValueError("tokens/lps length mismatch")
        if not ev["lps"]:          # 官方恒有第 0 个位置可取
            raise ValueError("empty generation")

    ratio = None
    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    elif threshold is not None:
        ratio, do_retrieve = _gate_ratio(no_retr["lp0"], threshold)
    else:
        do_retrieve = "[Retrieval]" in no_retr["text"]

    results = {}
    if mode != "always_retrieve":
        results["no_retrieval"] = {"pred": no_retr["text"], "score": None}

    pred = None
    scores = []
    if do_retrieve:
        if not evidences:
            raise ValueError("retrieval decided but no evidence")
        for i, ev in enumerate(evidences):
            sc = _evidence_score(ev, i, w_rel, w_sup, w_use, use_seqscore,
                                 use_groundness, use_utility)
            scores.append(sc)
            results["retrieval_%d" % i] = {"pred": ev["text"],
                                           "score": sc["final"]}
            pred = ev["text"]
    else:
        pred = second_text

    answer2score = None
    best_path = None
    answer_score = None
    if len(results) == 1:
        # 官方此分支直接返回第二次生成的 postprocess 结果：
        # 注意答案用的是 pred 变量（第二次生成），不是
        # results["no_retrieval"]["pred"]（第一次生成）
        answer = _postprocess(pred)
    elif closed:
        answer2score = {}
        for key, item in results.items():
            if key == "no_retrieval":
                continue
            ans = _postprocess_option(item["pred"])
            answer2score[ans] = answer2score.get(ans, 0) + item["score"]
        ranked = sorted(answer2score.items(), key=lambda kv: kv[1],
                        reverse=True)
        answer = ranked[0][0]
        answer_score = ranked[0][1]
    else:
        path2score = {key: item["score"] for key, item in results.items()
                      if key != "no_retrieval"}
        best_path = sorted(path2score.items(), key=lambda kv: kv[1],
                           reverse=True)[0][0]
        answer = results[best_path]["pred"]
        answer_score = results[best_path]["score"]

    return {"do_retrieve": do_retrieve,
            "retrieval_ratio": ratio,
            "result_keys": list(results.keys()),
            "evidence_scores": scores,
            "answer": answer,
            "best_path": best_path,
            "answer2score": answer2score,
            "answer_score": answer_score,
            "n_evidence": len(evidences)}
