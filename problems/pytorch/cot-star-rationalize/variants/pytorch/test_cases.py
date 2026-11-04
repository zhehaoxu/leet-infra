"""STaR（ezelikman/STaR）自举理性化账 -- 测试用例（期望值由官方
device_inference.py 源码 exec 沙箱实测钉死，rng 均为 random.Random(7)）."""
import random

EOT = "<|endoftext|>"


class ScriptedGen:
    """确定性生成替身：按 plan 逐次返回，同时记录 (contexts, hint)。"""

    def __init__(self, plan):
        self.plan = [list(p) for p in plan]
        self.calls = []
        self.i = 0

    def __call__(self, contexts, hint):
        outs = self.plan[self.i]
        self.i += 1
        self.calls.append({"contexts": list(contexts), "hint": hint,
                           "outputs": list(outs)})
        return outs


def gsm_ds(n):
    return [("gsm", {"question": f"q{i}", "answer": f"#### t{i}"})
            for i in range(n)]


def cqa_ds(n):
    return [("cqa", {"question": {"stem": f"s{i}",
                                  "choices": [{"label": "A", "text": "x"},
                                              {"label": "B", "text": "y"}]},
                     "answerKey": "a"}) for i in range(n)]


CQA_EX = {"question": {"stem": "Is a cat alive?",
                       "choices": [{"label": "A", "text": "yes"},
                                   {"label": "B", "text": "no"}]},
          "answerKey": "b"}

# ---------- mode="extract"：eval_output 三路抽取 ----------

def check_extract_cqa(solve):
    res = solve(mode="extract",
                outputs=["abc", "abc" + EOT + " junk", "ab", "axx\nsecond", ""],
                answers=["a", "a", "a", "a", "a"],
                base_contexts=["CTX0|", "CTX1|", "CTX2|", "CTX3|", "CTX4|"],
                classes=["cqa"] * 5, dataset_mode="cqa")
    # 首行抽取：多行输出只看第一行（第 4 条 axx\nsecond 命中）；
    # endoftext 子串剪裁；短首行 [-3] IndexError 静默跳过（total 仍计数）
    assert res["successful"] == [0, 1, 3], res
    assert res["train_examples"] == [
        "CTX0|abc" + EOT, "CTX1|abc" + EOT, "CTX3|axx" + EOT], res
    assert res["accuracy"] == {"cqa": {"accurate": 3, "total": 5}}, res
    # accuracy 可在外部 dict 上累计
    acc = {"cqa": {"accurate": 5, "total": 10}}
    res2 = solve(mode="extract", outputs=["abc"], answers=["a"],
                 base_contexts=["CTX0|"], classes=["cqa"], dataset_mode="cqa",
                 accuracy=acc)
    assert res2["accuracy"] == {"cqa": {"accurate": 6, "total": 11}}, res2
    assert res2["successful"] == [0]


def check_extract_gsm(solve):
    res = solve(mode="extract",
                outputs=["x = 3\n#### 42" + EOT, "I think 7",
                         "#### 1\n#### 2", "#### 7 ", "#### Forty",
                         "#### 9\n#### 9"],
                answers=["42", "42", "2", "7", "forty", "9"],
                base_contexts=["CTX0|", "CTX1|", "CTX2|", "CTX3|", "CTX4|",
                               "CTX5|"],
                classes=["gsm"] * 6, dataset_mode="gsm")
    # 无 #### 的输出预测空串判错不保存；首个 #### 行生效（双 #### 取首行截断）；
    # strip 与大小写不敏感
    assert res["successful"] == [0, 3, 4, 5], res
    assert res["train_examples"] == [
        "CTX0|x = 3\n#### 42" + EOT,
        "CTX3|#### 7 " + EOT,          # 保存的是未 strip 的截断串
        "CTX4|#### Forty" + EOT,
        "CTX5|#### 9" + EOT], res
    assert res["accuracy"] == {"gsm": {"accurate": 4, "total": 6}}, res


def check_extract_arith(solve):
    res = solve(mode="extract",
                outputs=["11 + 22\n= 33\n</scratch>\n33", "5\n" + EOT,
                         EOT + "\n9", "a\n</scratch>", "42" + EOT + "x", "12"],
                answers=["33", "5", "9", "33", "42", "33"],
                base_contexts=["CTX0|", "CTX1|", "CTX2|", "CTX3|", "CTX4|",
                               "CTX5|"],
                classes=["1", "1", "1", "2", "2", "2"],
                dataset_mode="arithmetic")
    # endoftext/</scratch> 是整行相等成员测试；endoftext 在第 0 行 -> -1 取最后一行；
    # </scratch> 行是末行 -> 跳过；跨行 endoftext 子串在 join 后再剪
    assert res["successful"] == [0, 1, 2, 4], res
    assert res["train_examples"] == [
        "CTX0|11 + 22\n= 33\n</scratch>\n33" + EOT,
        "CTX1|5" + EOT,
        "CTX2|" + EOT,                 # pi=-1 -> join 空前缀
        "CTX4|42" + EOT], res
    assert res["accuracy"] == {"1": {"accurate": 3, "total": 3},
                               "2": {"accurate": 1, "total": 3}}, res


def check_extract_arith_direct(solve):
    res = solve(mode="extract", outputs=["junk\n77", "55"],
                answers=["77", "77"], base_contexts=["CTX0|", "CTX1|"],
                classes=["1", "1"], dataset_mode="arithmetic", direct=True)
    # direct 且行数>1 取下标 1；单行仍取 0
    assert res["successful"] == [0], res
    assert res["train_examples"] == ["CTX0|junk\n77" + EOT], res


def check_extract_arith_fst_raise(solve):
    try:
        solve(mode="extract", outputs=["2"], answers=["2"],
              base_contexts=["CTX0|"], classes=["1"],
              dataset_mode="arithmetic", few_shot_train=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("arithmetic + few_shot_train must raise")


def check_extract_length_mismatch(solve):
    try:
        solve(mode="extract", outputs=["a", "b"], answers=["a"],
              base_contexts=["C|", "C|"], classes=["cqa", "cqa"],
              dataset_mode="cqa")
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")


# ---------- mode="score"：get_score ----------

def check_score(solve):
    assert solve(mode="score", counts={"accurate": 0, "total": 0}) == 0
    assert solve(mode="score", counts={"accurate": 3, "total": 8}) == 0.375


# ---------- mode="context"：question_to_context ----------

def check_context_cqa(solve):
    assert solve(mode="context", example=CQA_EX, dataset_mode="cqa") == \
        "Q: Is a cat alive?\nAnswer Choices:\n(a) yes\n(b) no\nA:"
    assert solve(mode="context", example=CQA_EX, dataset_mode="cqa",
                 hint=True) == \
        "Q: Is a cat alive?\nAnswer Choices:\n(a) yes\n(b) no (CORRECT)\nA:"
    # answerKey 大小写不敏感匹配
    up = {"question": CQA_EX["question"], "answerKey": "B"}
    assert solve(mode="context", example=up, dataset_mode="cqa", hint=True) == \
        "Q: Is a cat alive?\nAnswer Choices:\n(a) yes\n(b) no (CORRECT)\nA:"


def check_context_gsm(solve):
    ex = {"question": "2+2?", "answer": "#### 4"}
    assert solve(mode="context", example=ex, dataset_mode="gsm") == "Q: 2+2?\nA:"
    assert solve(mode="context", example=ex, dataset_mode="gsm", hint=True) == \
        "Q: 2+2? (#### 4)\nA:"


def check_context_arith(solve):
    ex = "1 + 1\nTarget: 2\n2 + 2\nTarget: 4"
    # 相邻切片对逐对拼 seg + 'Target:'，末段（真答案）被丢弃
    assert solve(mode="context", example=ex, dataset_mode="arithmetic") == \
        "1 + 1\nTarget: 2\n2 + 2\nTarget:"
    hint_ex = ("scr1\nscr2\n</scratch>\nkeep1\n"
               "Target: n1a\nn1b\nn1c\nn1d\nn1e\n"
               "Target: n2a\nn2b\nn2c\nn2d\nn2e\n"
               "Target: t9a\nt9b\nt9c\nt9d\nt9e")
    # direct 只保留 </scratch> 后半段；hint 追加下一段倒数第 5 行
    assert solve(mode="context", example=hint_ex, dataset_mode="arithmetic",
                 hint=True, direct=True) == \
        ("\nkeep1\nTarget: n1b n1a\nn1b\nn1c\nn1d\nn1e\n"
         "Target: n2b n2a\nn2b\nn2c\nn2d\nn2e\nTarget:  t9a")
    assert solve(mode="context", example=hint_ex,
                 dataset_mode="arithmetic") == \
        ("scr1\nscr2\n</scratch>\nkeep1\nTarget: n1a\nn1b\nn1c\nn1d\nn1e\n"
         "Target: n2a\nn2b\nn2c\nn2d\nn2e\nTarget:")
    assert solve(mode="context", example="no targets here",
                 dataset_mode="arithmetic") == ""


# ---------- mode="eval"：eval_examples 分批 + hint 池理性化 ----------

def check_eval_leftover(solve):
    gen = ScriptedGen([["#### t0", "#### t1"]])
    res = solve(mode="eval", dataset=gsm_ds(2), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=[], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3)
    # 2 < batch_size：整批永不评测
    assert res["accuracy"] == {} and res["train_examples"] == [], res
    assert gen.calls == [], gen.calls


def check_eval_rationalize_noflush(solve):
    gen = ScriptedGen([["#### t0", "#### nope", "#### t2"],
                       ["#### t3", "#### t4", "#### zz"]])
    res = solve(mode="eval", dataset=gsm_ds(7), dataset_mode="gsm",
                prompts=["P1", "P2"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True,
                p_rationalization=1.0)
    # batch1/batch2 各 1 miss -> 池 2 < 3 不 flush；第 7 个样本永不评测
    assert res["accuracy"] == {"gsm": {"accurate": 4, "total": 6}}, res
    assert res["train_examples"] == [
        "Q: q0\nA:#### t0" + EOT, "Q: q2\nA:#### t2" + EOT,
        "Q: q3\nA:#### t3" + EOT, "Q: q4\nA:#### t4" + EOT], res
    assert [c["hint"] for c in gen.calls] == [False, False], gen.calls
    assert gen.calls[0]["contexts"] == [
        "P1\n\nP2\n\nQ: q0\nA:", "P1\n\nP2\n\nQ: q1\nA:",
        "P1\n\nP2\n\nQ: q2\nA:"], gen.calls


def check_eval_hint_save(solve):
    gen = ScriptedGen([["nope", "nope", "nope"], ["aaa", "aaa", "aaa"]])
    res = solve(mode="eval", dataset=cqa_ds(3), dataset_mode="cqa",
                prompts=None, hint_prompts=None, gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True,
                p_rationalization=1.0, p_show_hint_save=1.0)
    # p_show_hint_save=1.0：hint 批保存用 base（带 (CORRECT)），accuracy 进 cqa_r 桶
    assert res["accuracy"] == {"cqa": {"accurate": 0, "total": 3},
                               "cqa_r": {"accurate": 3, "total": 3}}, res
    assert res["train_examples"] == [
        "Q: s0\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:aaa" + EOT,
        "Q: s1\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:aaa" + EOT,
        "Q: s2\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:aaa" + EOT], res
    assert [c["hint"] for c in gen.calls] == [False, True], gen.calls
    assert gen.calls[1]["contexts"] == [
        "Q: s0\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:",
        "Q: s1\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:",
        "Q: s2\nAnswer Choices:\n(a) x (CORRECT)\n(b) y\nA:"], gen.calls


def check_eval_p_rat_zero(solve):
    gen = ScriptedGen([["#### nope"] * 3, ["#### nope"] * 3])
    res = solve(mode="eval", dataset=gsm_ds(7), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True,
                p_rationalization=0.0)
    # p=0：miss 全不入池，永不 flush
    assert res["accuracy"] == {"gsm": {"accurate": 0, "total": 6}}, res
    assert res["train_examples"] == [] and len(gen.calls) == 2, (res, gen.calls)


def check_eval_few_shot_train(solve):
    gen = ScriptedGen([["aaa", "aaa", "aaa"]])
    res = solve(mode="eval", dataset=cqa_ds(3), dataset_mode="cqa",
                prompts=["F1", "F2"], hint_prompts=[], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, few_shot_train=True,
                train_prompts=["T1", "T2"])
    # 保存样本带 train_prompts 前缀；输入上下文仍用 prompts
    assert res["train_examples"] == [
        "T1\n\nT2\n\nQ: s0\nAnswer Choices:\n(a) x\n(b) y\nA:aaa" + EOT,
        "T1\n\nT2\n\nQ: s1\nAnswer Choices:\n(a) x\n(b) y\nA:aaa" + EOT,
        "T1\n\nT2\n\nQ: s2\nAnswer Choices:\n(a) x\n(b) y\nA:aaa" + EOT], res
    assert gen.calls[0]["contexts"] == [
        "F1\n\nF2\n\nQ: s0\nAnswer Choices:\n(a) x\n(b) y\nA:",
        "F1\n\nF2\n\nQ: s1\nAnswer Choices:\n(a) x\n(b) y\nA:",
        "F1\n\nF2\n\nQ: s2\nAnswer Choices:\n(a) x\n(b) y\nA:"], gen.calls
    try:
        solve(mode="eval", dataset=[("1", "5\n5\n5")], dataset_mode="arithmetic",
              prompts=["P"], hint_prompts=[], gen_fn=ScriptedGen([["x"]]),
              rng=random.Random(7), eval_batch_size=1, few_shot_train=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("eval + arithmetic + few_shot_train must raise")


def check_eval_arith(solve):
    ds = [("1", "5 + 5\nTARGETLINE\nTarget: x\n"),
          ("1", "5 + 5\nTARGETLINE\nTarget: x\n"),
          ("2", "7 + 7\nNOLINE\nTarget: y\n")]
    gen = ScriptedGen([["TARGETLINE", "TARGETLINE", "zzz"]])
    res = solve(mode="eval", dataset=ds, dataset_mode="arithmetic",
                prompts=["A1"], hint_prompts=[], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3)
    # arithmetic：few-shot 被 base.split('\n\n')[:-1] 覆盖为 []，前缀只剩 '\n\n'
    assert gen.calls[0]["contexts"] == [
        "\n\n5 + 5\nTARGETLINE\nTarget:",
        "\n\n5 + 5\nTARGETLINE\nTarget:",
        "\n\n7 + 7\nNOLINE\nTarget:"], gen.calls
    assert res["accuracy"] == {"1": {"accurate": 2, "total": 2},
                               "2": {"accurate": 0, "total": 1}}, res
    assert res["train_examples"] == [
        "5 + 5\nTARGETLINE\nTarget:TARGETLINE" + EOT,
        "5 + 5\nTARGETLINE\nTarget:TARGETLINE" + EOT], res


def check_eval_rat_false_pool(solve):
    gen = ScriptedGen([["#### nope"] * 3])
    res = solve(mode="eval", dataset=gsm_ds(3), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=False,
                p_rationalization=1.0)
    # miss 照常入池（准入不受 rationalize 门控），但永不 flush
    assert res["accuracy"] == {"gsm": {"accurate": 0, "total": 3}}, res
    assert res["train_examples"] == [] and len(gen.calls) == 1, (res, gen.calls)
    assert gen.calls[0]["contexts"] == [
        "P1\n\nQ: q0\nA:", "P1\n\nQ: q1\nA:", "P1\n\nQ: q2\nA:"], gen.calls


def check_eval_cqa_r_bucket(solve):
    ds = [("cqa", dict(CQA_EX, answerKey="a")) for _ in range(3)]
    gen = ScriptedGen([["nope", "nope", "nope"], ["aaa", "nope", "aaa"]])
    res = solve(mode="eval", dataset=ds, dataset_mode="cqa",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True)
    # hint 批 key+'_r' 独立分桶；保存用 hintless（无 (CORRECT)）
    assert res["accuracy"] == {"cqa": {"accurate": 0, "total": 3},
                               "cqa_r": {"accurate": 2, "total": 3}}, res
    assert res["train_examples"] == [
        "Q: Is a cat alive?\nAnswer Choices:\n(a) yes\n(b) no\nA:aaa" + EOT,
        "Q: Is a cat alive?\nAnswer Choices:\n(a) yes\n(b) no\nA:aaa" + EOT], res
    assert gen.calls[1]["contexts"] == [
        "H1\n\nQ: Is a cat alive?\nAnswer Choices:\n(a) yes (CORRECT)\n(b) no\nA:",
        "H1\n\nQ: Is a cat alive?\nAnswer Choices:\n(a) yes (CORRECT)\n(b) no\nA:",
        "H1\n\nQ: Is a cat alive?\nAnswer Choices:\n(a) yes (CORRECT)\n(b) no\nA:"], gen.calls


def check_eval_double_flush(solve):
    gen = ScriptedGen([["#### nope"] * 3, ["#### nope"] * 3,
                       ["#### t0", "#### t1", "#### t2"],
                       ["#### nope"] * 3])
    res = solve(mode="eval", dataset=gsm_ds(6), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True)
    # 批1 全 miss -> 池满 flush1（全 miss）；批2 全 miss -> flush2
    assert res["accuracy"] == {"gsm": {"accurate": 0, "total": 6},
                               "gsm_r": {"accurate": 0, "total": 6}}, res
    assert [c["hint"] for c in gen.calls] == [False, True, False, True], gen.calls
    assert gen.calls[1]["contexts"] == [
        "H1\n\nQ: q0 (#### t0)\nA:", "H1\n\nQ: q1 (#### t1)\nA:",
        "H1\n\nQ: q2 (#### t2)\nA:"], gen.calls


def check_eval_p_rat_half(solve):
    gen = ScriptedGen([["#### nope"] * 3, ["#### nope"] * 3,
                       ["#### nope"] * 3,
                       ["#### t0", "#### t1", "#### t2"],
                       ["#### t3", "#### t4", "#### t5"]])
    res = solve(mode="eval", dataset=gsm_ds(9), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True,
                p_rationalization=0.5)
    # 撒币结果（Random(7)）：批1 入池 q0/q2、批2 入池 q3/q4 -> 池4? 官方实测
    # 在批2 后池 >=3 flush 前 3 个（q0,q2,q3），批3 后池 q4,q5,q7 flush
    assert res["accuracy"] == {"gsm": {"accurate": 0, "total": 9},
                               "gsm_r": {"accurate": 0, "total": 6}}, res
    assert res["train_examples"] == [], res
    assert gen.calls[2]["contexts"] == [
        "H1\n\nQ: q0 (#### t0)\nA:", "H1\n\nQ: q2 (#### t2)\nA:",
        "H1\n\nQ: q3 (#### t3)\nA:"], gen.calls
    assert gen.calls[4]["contexts"] == [
        "H1\n\nQ: q4 (#### t4)\nA:", "H1\n\nQ: q5 (#### t5)\nA:",
        "H1\n\nQ: q7 (#### t7)\nA:"], gen.calls


def check_eval_carry_over(solve):
    gen = ScriptedGen([["#### t0", "#### nope", "#### nope"],
                       ["#### t3", "#### nope", "#### nope"],
                       ["#### t6", "#### nope", "#### nope"],
                       ["#### t1", "#### t2", "#### t4"],
                       ["#### t5", "#### t7", "#### t8"]])
    res = solve(mode="eval", dataset=gsm_ds(9), dataset_mode="gsm",
                prompts=["P1"], hint_prompts=["H1"], gen_fn=gen,
                rng=random.Random(7), eval_batch_size=3, rationalize=True)
    # 批2 后池 4 -> flush 3 留 1（carry-over）；批3 后池 3 -> flush
    assert res["accuracy"] == {"gsm": {"accurate": 2, "total": 9},
                               "gsm_r": {"accurate": 1, "total": 6}}, res
    assert res["train_examples"] == [
        "Q: q0\nA:#### t0" + EOT, "Q: q3\nA:#### t3" + EOT,
        "Q: q5\nA:#### t5" + EOT], res
    assert gen.calls[2]["contexts"] == [
        "H1\n\nQ: q1 (#### t1)\nA:", "H1\n\nQ: q2 (#### t2)\nA:",
        "H1\n\nQ: q4 (#### t4)\nA:"], gen.calls
    assert gen.calls[4]["contexts"] == [
        "H1\n\nQ: q5 (#### t5)\nA:", "H1\n\nQ: q6 (#### t6)\nA:",
        "H1\n\nQ: q7 (#### t7)\nA:"], gen.calls


# ---------- 参数校验 ----------

def check_args_validation(solve):
    bad = [("mode", lambda: solve(mode="nope")),
           ("dataset_mode", lambda: solve(mode="extract", outputs=["a"],
                                          answers=["a"], base_contexts=["C"],
                                          classes=["c"], dataset_mode="mmlu")),
           ("score_keys", lambda: solve(mode="score", counts={"accurate": 1})),
           ("score_type", lambda: solve(mode="score",
                                        counts={"accurate": True, "total": 2})),
           ("cqa_dict", lambda: solve(mode="context", example="str",
                                      dataset_mode="cqa")),
           ("arith_dict", lambda: solve(mode="context", example={"q": 1},
                                        dataset_mode="arithmetic")),
           ("batch_size", lambda: solve(mode="eval", dataset=gsm_ds(3),
                                        dataset_mode="gsm", prompts=["P"],
                                        hint_prompts=[],
                                        gen_fn=ScriptedGen([["a"]]),
                                        rng=random.Random(7),
                                        eval_batch_size=0)),
           ("p_range", lambda: solve(mode="eval", dataset=gsm_ds(3),
                                     dataset_mode="gsm", prompts=["P"],
                                     hint_prompts=["H1"],
                                     gen_fn=ScriptedGen([["a", "a", "a"]]),
                                     rng=random.Random(7), eval_batch_size=3,
                                     rationalize=True,
                                     p_rationalization=1.5)),
           ("gen_fn", lambda: solve(mode="eval", dataset=gsm_ds(3),
                                    dataset_mode="gsm", prompts=["P"],
                                    hint_prompts=[], gen_fn="not callable",
                                    rng=random.Random(7), eval_batch_size=3)),
           ("rng", lambda: solve(mode="eval", dataset=gsm_ds(3),
                                 dataset_mode="gsm", prompts=["P"],
                                 hint_prompts=[],
                                 gen_fn=ScriptedGen([["a"]]),
                                 rng="not random", eval_batch_size=3)),
           ("dataset_pair", lambda: solve(mode="eval", dataset=["bad"],
                                          dataset_mode="gsm", prompts=["P"],
                                          hint_prompts=[],
                                          gen_fn=ScriptedGen([["a"]]),
                                          rng=random.Random(7),
                                          eval_batch_size=3))]
    for name, fn in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"{name} must raise ValueError")
