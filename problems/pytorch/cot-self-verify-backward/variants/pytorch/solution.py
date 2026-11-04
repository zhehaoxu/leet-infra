"""Self-Verification 的后向验证账（候选收集 + 数字挖空 + verifier 评分）。

官方 WENGSYX/Self-Verification 的确定性核心（main_verifier.py +
utils/question_turn_decalrative）：把每个候选答案回填进问题做「后向验证」
——从问题文本抽出全部数字，逐个把该数字替换为 'X' 构造子问题，verifier
对每个子问题给 True/False 判定，判定与挖空前原数字一致者加分，总分最高
的候选胜出。（GPT 采样、answer_cleansing 与 verifier 模型不入判题域：
采样串与判定串均显式入参。）

1. 候选收集（main_verifier L140-157）：逐个采样串尝试 `str(float(x))`
   规范化（数值串统一 "5"→"5.0"，非数值原样），非空且**未见过的**才
   append（首现序去重）；
2. declarative 改写缓存（utils L854-864）：declarative 非空且含
   answers[0] 时，把 **最后一次**出现的 answers[0] 替换为当前候选
   （官方用反向 find 定位，等价 rfind 替换）；declarative 为空或不含
   answers[0] 且无上游改写时，fallback 模板
   `"{question} The answer is {answer}."`（官方 GPT 重生成不入判题域，
   fallback 直接采用）；
3. 数字挖空子问题（utils L866-892）：
   a. text 先做 `replace(',', '.')`（官方把逗号当句点用）；
   b. `ts = text[:len(text) - text[::-1].find('.')]`（最后一个句号处
      截断，含句号；`question` 尾段在数值路径未被使用）；
   c. `numbers = findall(r'-?\\d+\\.?\\d*', ts)`（含负号与小数点）；
   d. 对每个数字：遍历 `ts.split('.')` 句段，**每个**含该数字的句段
      都替换（官方 number_find 置位缺失，恒 False——忠实复刻；替换
      是字符列表切片赋值 `"'X'"`，首现位置），未替换句段原样补 '.'，
      再 `replace('..', '.')`（尾空段产生双句点的官方修正）；
   e. new_ts 首空格去掉、尾无空格补一个空格、`replace(' .', '.')`；
   f. 子问题串 = `'"' + new_ts + declarative + '"'`；**最多取前 3 个**
      数字（官方 sentences[:3] / ans[:3]）；
4. verifier 评分（main_verifier L190-208）：每个子问题的判定串逐个
   try float 转换；期望值 try float(挖空数字) 转换，都失败 score=0；
   `score_d = sum(v == expected)`，候选总分 = Σ_d score_d；
5. 选择与判定（L213-259）：候选唯一时 scores = {0: 1}（不构造子问题、
   不评分）；否则 scores 按候选首现序累加，`argmax` **平局取最前**
   （np.argmax 语义）；is_true = 每个候选（规范化后）是否等于规范化
   GT；chosen 对/错由 chosen == GT_norm 决定。

返回 `dict(answers, sentences, sub_numbers, scores, chosen_idx,
chosen_answer, is_true, chosen_correct)`。
"""

import re

_NO_SOLUTION = ""


def _normalize(s):
    """官方 main_verifier L143-146：数值串统一 str(float(x))，否则原样。"""
    try:
        return str(float(s))
    except Exception:
        return s


def _make_declarative(question, answer):
    return "{} The answer is {}.".format(question, answer)


def _subquestions(text, answer, answers_0, declarative):
    """官方 question_turn_decalrative 数值路径（L835-892）。"""
    text = text.replace(",", ".")
    position_fullstop = text[::-1].find(".")
    question = text[len(text) - position_fullstop:]
    ts = text[:len(text) - position_fullstop]

    if declarative == "":
        declarative = _make_declarative(question, answer)
    else:
        if answers_0 in declarative:
            # 官方反向定位拼接：替换最后一次出现的 answers_0
            rev = declarative[::-1].find(answers_0[::-1])
            start = len(declarative) - rev - len(answers_0)
            declarative = (declarative[:start] + answer
                           + declarative[start + len(answers_0):])
        else:
            declarative = _make_declarative(question, answer)

    numbers = [s for s in re.findall(r"-?\d+\.?\d*", ts)]
    sentences, ans = [], []
    for nqn in numbers:
        new_ts = ""
        for i in ts.split("."):
            if nqn in i:
                chars = [p for p in i]
                pos = i.find(nqn)
                chars[pos:pos + len(nqn)] = "'X'"
                # 官方 new_question.replace(' .', '.') 不接收结果（死代码）
                new_ts += "".join(chars) + "."
            else:
                new_ts += i + "."
        new_ts = new_ts.replace("..", ".")
        if new_ts[0] == " ":
            new_ts = new_ts[1:]
        if new_ts[-1] != " ":
            new_ts += " "
        new_ts = new_ts.replace(" .", ".")
        sentences.append('"' + new_ts + declarative + '"')
        ans.append(nqn)
    return sentences[:3], ans[:3], declarative


def _score_subquestion(verdicts, expected_raw):
    """官方 L190-208：判定串逐个 try float，期望值 try float，失败 0 分。"""
    converted = []
    for v in verdicts:
        try:
            converted.append(float(v))
        except Exception:
            converted.append(v)
    try:
        expected = float(expected_raw)
    except Exception:
        try:
            expected = expected_raw
        except Exception:
            return 0
    try:
        return sum(1 for v in converted if v == expected)
    except Exception:
        return 0


def solve(question, sampled_answers, verifier_outputs, ground_truth,
          declarative=""):
    if not isinstance(question, str) or not question:
        raise ValueError("question 必须是非空字符串")
    if not isinstance(sampled_answers, list) or not sampled_answers:
        raise ValueError("sampled_answers 必须是非空 list")
    for i, s in enumerate(sampled_answers):
        if not isinstance(s, str):
            raise ValueError(f"sampled_answers[{i}] 必须是字符串")
    if not isinstance(verifier_outputs, list):
        raise ValueError("verifier_outputs 必须是 list")
    for i, vs in enumerate(verifier_outputs):
        if not isinstance(vs, list):
            raise ValueError(f"verifier_outputs[{i}] 必须是 list")
        for j, v in enumerate(vs):
            if not isinstance(v, str):
                raise ValueError(f"verifier_outputs[{i}][{j}] 必须是字符串")
    if not isinstance(ground_truth, str):
        raise ValueError("ground_truth 必须是字符串")
    if not isinstance(declarative, str):
        raise ValueError("declarative 必须是字符串")

    # 1. 候选收集：float 规范化 + 非空 + 首现去重
    answers = []
    for s in sampled_answers:
        item = _normalize(s)
        if item != _NO_SOLUTION and item not in answers:
            answers.append(item)
    gt_norm = _normalize(ground_truth)

    # 4. 单候选：不验证直接 scores={0:1}
    scores = {}
    sentences = []
    sub_numbers = []
    if len(answers) == 1:
        scores = {0: 1}
    else:
        for a_idx, ans_a in enumerate(answers):
            decl, ans_list, _ = _subquestions(
                question, ans_a, answers[0], declarative)
            total = 0
            for d in range(len(decl)):
                verdicts = verifier_outputs[a_idx * 3 + d] \
                    if a_idx * 3 + d < len(verifier_outputs) else []
                total += _score_subquestion(verdicts, ans_list[d])
            scores[a_idx] = total
            sentences.append(decl)
            sub_numbers.append(ans_list)

    # 5. argmax 平局取最前（np.argmax 语义）
    chosen_idx = max(scores, key=lambda k: (scores[k], -k))
    chosen_answer = answers[chosen_idx]
    is_true = [a == gt_norm for a in answers]
    chosen_correct = chosen_answer == gt_norm
    return {
        "answers": answers,
        "sentences": sentences,
        "sub_numbers": sub_numbers,
        "scores": scores,
        "chosen_idx": chosen_idx,
        "chosen_answer": chosen_answer,
        "is_true": is_true,
        "chosen_correct": chosen_correct,
    }
