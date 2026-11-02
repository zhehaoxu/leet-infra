"""self-instruct：合成指令的后处理过滤、ROUGE 阈值去重与提示拼装。"""

import math
import re
import string

# 与 yizhongw/self-instruct 的 bootstrap_instructions.py 逐字同源
SPLIT = re.compile(r"\n\d+\s?\. ")
WS = re.compile(r"\s+")
KEYWORDS = ["image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to"]
KEYWORD_RES = [(w, re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE))
               for w in KEYWORDS]
HEAD_CLF = ("Come up with a series of classification tasks. "
            "Try to specify the possible output labels when possible.\n")
HEAD_GEN = "Come up with a series of tasks:\n"


def _normalize(seg):
    """官方两步：先 `\\s+` 压成单空格再 strip，然后 `.strip().capitalize()`。"""
    return WS.sub(" ", seg).strip().strip().capitalize()


def _reason(inst):
    """按官方顺序给出第一条命中的拒绝理由；None 表示通过。"""
    if inst == "":
        return "empty"
    words = inst.split()
    if len(words) <= 3:
        return "too_short"
    if len(words) > 150:
        return "too_long"
    for _, rx in KEYWORD_RES:
        if rx.search(inst):
            return "keyword"
    if inst.startswith("Write a program"):
        return "write_a_program"
    if inst[0] in string.punctuation:
        return "punct"
    if not inst[0].isascii():
        return "non_ascii"
    return None


def _encode_prompt(instructions, classification):
    prompt = HEAD_CLF if classification else HEAD_GEN
    for idx, instruction in enumerate(instructions):
        text = WS.sub(" ", instruction).strip().rstrip(":")
        prompt += f"{idx + 1}. {text}\n"
    prompt += f"{len(instructions) + 1}."
    return prompt


def solve(response, seed_instructions, sims, finish_reason="stop",
          threshold=0.7, top_k=10, prompt_instructions=None,
          classification=False):
    if response is not None and not isinstance(response, str):
        raise ValueError("response 必须是字符串或 None")
    if not isinstance(finish_reason, str):
        raise ValueError("finish_reason 必须是字符串")
    if not isinstance(seed_instructions, (list, tuple)):
        raise ValueError("seed_instructions 必须是列表")
    for s in seed_instructions:
        if not isinstance(s, str):
            raise ValueError("seed_instructions 每项必须是字符串")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold 必须是实数")
    th = float(threshold)
    if not math.isfinite(th) or th < 0 or th > 1:
        raise ValueError("threshold 必须在 [0, 1] 内")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k 必须是 >=1 的整数")
    if sims is not None and not isinstance(sims, (list, tuple)):
        raise ValueError("sims 必须是列表或 None")
    if prompt_instructions is not None:
        if not isinstance(prompt_instructions, (list, tuple)):
            raise ValueError("prompt_instructions 必须是列表或 None")
        for s in prompt_instructions:
            if not isinstance(s, str):
                raise ValueError("prompt_instructions 每项必须是字符串")
    if not isinstance(classification, bool):
        raise ValueError("classification 必须是 bool")

    # 1. 截断响应直接放弃（官方 finish_reason == "length" 时返回空）
    segments = []
    if response is not None and finish_reason != "length":
        for seg in SPLIT.split(response):
            inst = _normalize(seg)
            reason = _reason(inst)
            segments.append(
                {"raw": seg, "instruction": inst, "kept": reason is None,
                 "reason": reason}
            )

    candidates = [r["instruction"] for r in segments if r["kept"]]
    if sims is None:  # 不提供相似度：跳过去重，只跑第一层
        sim_vectors = None
    else:
        sim_vectors = [list(v) for v in sims]
    if sim_vectors is not None:
        for v in sim_vectors:
            for x in v:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise ValueError("sims 内每项必须是实数")
                if not math.isfinite(float(x)):
                    raise ValueError("sims 不得含 nan / inf")
        if len(sim_vectors) != len(candidates):
            raise ValueError("sims 条数必须等于通过第一层的候选数")

    # 2. 阈值去重：池子 = 种子 + 已接受机器指令，逐条顺序增长
    pool = list(seed_instructions)
    dedup, instructions = [], []
    if sim_vectors is None:
        for inst in candidates:
            dedup.append({"instruction": inst, "accepted": True,
                          "max_similarity": None, "avg_similarity": 0.0,
                          "most_similar": {}})
            instructions.append(inst)
        sim_vectors = []
    for inst, vec in zip(candidates, sim_vectors):
        if len(vec) != len(pool):
            raise ValueError(
                "第 %d 条候选的相似度向量长度必须等于当前池大小"
                % len(dedup))
        best = max(vec) if vec else None
        accepted = not (best is not None and best > th)
        avg = (sum(vec) / len(vec)) if vec else 0.0
        most_similar = {}
        if accepted:
            order = sorted(range(len(vec)), key=lambda i: (vec[i], i))
            for i in order[-top_k:][::-1]:
                most_similar[pool[i]] = vec[i]
            pool.append(inst)
            instructions.append(inst)
        dedup.append(
            {"instruction": inst, "accepted": accepted,
             "max_similarity": best, "avg_similarity": avg,
             "most_similar": most_similar}
        )

    prompt = None
    if prompt_instructions is not None:
        prompt = _encode_prompt(list(prompt_instructions), classification)

    return {
        "segments": segments,
        "candidates": candidates,
        "dedup": dedup,
        "instructions": instructions,
        "prompt": prompt,
    }
