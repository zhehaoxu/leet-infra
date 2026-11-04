"""STaR（arXiv:2203.14465, ezelikman/STaR）自举理性化账。

官方 device_inference.py 把「生成 -> 抽取答案 -> 判对错 -> 答对样本写回
训练集」组织成 eval_output / eval_examples 两个确定性账本：三种数据集
（cqa / gsm / arithmetic）各有一条钉死的答案抽取链，判对的输出按原始
行拼回 base context 组装新训练样本；eval_examples 再把数据集按
eval_batch_size 分批，答错样本以概率 p_rationalization 进 hint 池，
池满一批就以 key+'_r' 重命名用带提示（hint）上下文重评（理性化）。
本题把 LLM 生成全部显式入参（outputs 已解码字符串 / gen_fn 可调用），
考纯账本。

solve(mode=...) 四个入口（官方逐条钉死）：

    mode="extract"（eval_output）：outputs + answers + base_contexts +
        classes + dataset_mode -> {"successful", "train_examples",
        "accuracy"}。accuracy 按 class 分桶累计 {'accurate', 'total'}，
        total 在抽取 try 之前自增（短输出/IndexError 的样本也计数）。
        抽取链：
          cqa：取首行 -> 剪 <|endoftext|> 子串 -> 预测 = 首行[-3] 字符
             （len<3 抛 IndexError，官方 except IndexError 静默跳过）；
          gsm：扫到首条含 '####' 的行，把输出截断到该行（含），再剪
              endoftext，预测 = 截断串 split('####')[-1].strip()；扫不到
              '####' 则预测为空串（判错不保存，不会写回）；
          arithmetic：output_numbers 仍是行列表——'<|endoftext|>' 与
              '</scratch>' 都是**整行相等**的成员测试；endoftext 行下标-1
              为预测行（下标 0 时得 -1 -> 取最后一行，Python 负索引）；
              '</scratch>' 行下标+1 == len 则跳过该样本；否则 direct 且
              行数>1 取下标 1、否则 0。保存时 join(行[:预测行+1]) 再剪
              endoftext 子串。few_shot_train 且 arithmetic 且判对 ->
              NotImplementedError（官方 raise，不被 except IndexError 吞）。
        判对大小写不敏感（两侧 .lower()）；保存样本 = base_context +
        抽取用中间串 + '<|endoftext|>'（cqa 用首行、gsm 用截断串、
        arithmetic 用截到预测行的 join）。

    mode="score"（get_score）：counts {'accurate','total'} -> total==0
        返回 int 0，否则 accurate/total。

    mode="context"（question_to_context）：三数据集上下文组装。
          cqa：'Q: {stem}\\nAnswer Choices:\\n' + 逐选项
              '({label.lower()}) {text}\\n'（hint 且 label.lower() ==
              answerKey.lower() 的选项追加 ' (CORRECT)'）+ 'A:'；
          gsm：'Q: {question}'（hint 追加 ' ({answer})'）+ '\\nA:'；
          arithmetic：example 按 'Target:' 切片 zip 相邻对，每对贡献
              seg + 'Target:'（direct 且 seg 含 '</scratch>' 时只取
              split('</scratch>')[-1]），hint 再追加
              ' ' + 下一段 split('\\n')[-5]；无 'Target:' -> 空串。

    mode="eval"（eval_examples）：分批 + hint 池理性化循环。
        dataset 是 (key, example) 对列表；每凑满 eval_batch_size 评一批
        （不足最后一批**永不评测**），批内逐样本：
          - 上下文：hintless（无提示）与 base（hint 版）各算一份；
            arithmetic 时 few-shot 改从 base.split('\\n\\n')[:-1] 取、
            两者取 split('\\n\\n')[-1]；rng.random() < p_show_hint_save
            则保存用 hintless 换成 base；few_shot_train 时保存用
            '\\n\\n'.join(train_prompts) + '\\n\\n' 前缀（arithmetic 直接
            NotImplementedError）；输入上下文 = few-shot join 前缀（
            prompts=None 则空）+ base；
          - target：arithmetic 取倒数第三行（行数<3 为 'invalid'）、cqa
            取 answerKey、gsm 取 answer.split('#### ')[-1]；
        批评完后：答错样本以 rng.random() < p_rationalization 进 hint
        池；rationalize 且池 >= eval_batch_size 立即取前一批（key 追加
        '_r' -> accuracy 独立桶）以 hint 上下文重评，剩余留在池里。
        返回 {"accuracy", "train_examples"}；rng 需为 random.Random 实例，
        翻硬币顺序：批内逐样本 p_show_hint_save 先于批后逐 miss 的
        p_rationalization。
"""
import random

ENDOFTEXT = "<|endoftext|>"
DATASET_MODES = ("cqa", "gsm", "arithmetic")


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _check_str_list(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"each item in {name} must be a str")


def _check_dataset_mode(dataset_mode):
    if dataset_mode not in DATASET_MODES:
        raise ValueError(
            f"dataset_mode must be one of {DATASET_MODES}, got {dataset_mode!r}")


def _eval_output(outputs, answers, base_contexts, classes, dataset_mode,
                 accuracy=None, direct=False, few_shot_train=False,
                 endoftext=ENDOFTEXT):
    """官方 eval_output：三路抽取 + 判对过滤 + 训练样本组装。"""
    _check_str_list(outputs, "outputs")
    _check_str_list(answers, "answers")
    _check_str_list(base_contexts, "base_contexts")
    _check_str_list(classes, "classes")
    if not (len(outputs) == len(answers) == len(base_contexts) == len(classes)):
        raise ValueError("outputs/answers/base_contexts/classes must have equal length")
    _check_dataset_mode(dataset_mode)
    _check_bool(direct, "direct")
    _check_bool(few_shot_train, "few_shot_train")
    if not isinstance(endoftext, str):
        raise ValueError("endoftext must be a str")
    if accuracy is None:
        accuracy = {}
    if not isinstance(accuracy, dict):
        raise ValueError("accuracy must be a dict")
    successful_examples = []
    train_examples = []
    for idx, cur_output in enumerate(outputs):
        target = answers[idx]
        cur_base_context = base_contexts[idx]
        example_class = classes[idx]
        output_numbers = cur_output.split('\n')
        if example_class not in accuracy:
            accuracy[example_class] = {'accurate': 0, 'total': 0}
        accuracy[example_class]['total'] += 1
        if len(output_numbers) == 0:
            continue
        try:
            if dataset_mode == "cqa":
                output_numbers = output_numbers[0]
                if ENDOFTEXT in output_numbers:
                    output_numbers = output_numbers.split(ENDOFTEXT)[0]
                output_prediction = output_numbers[-3]
            elif dataset_mode == "gsm":
                output_prediction = ""
                for line_idx, line in enumerate(output_numbers):
                    if "####" in line:
                        output_numbers = "\n".join(output_numbers[:line_idx + 1])
                        if ENDOFTEXT in output_numbers:
                            output_numbers = output_numbers.split(ENDOFTEXT)[0]
                        output_prediction = output_numbers.split("####")[-1].strip()
                        break
            elif dataset_mode == "arithmetic":
                if len(output_numbers) == 0:
                    continue
                elif ENDOFTEXT in output_numbers:
                    prediction_index = output_numbers.index(ENDOFTEXT) - 1
                elif "</scratch>" in output_numbers:
                    prediction_index = output_numbers.index("</scratch>") + 1
                    if prediction_index == len(output_numbers):
                        continue
                else:
                    if direct and len(output_numbers) > 1:
                        prediction_index = 1
                    else:
                        prediction_index = 0
                output_prediction = output_numbers[prediction_index]

            if ENDOFTEXT in output_prediction:
                output_prediction = output_prediction.split(ENDOFTEXT)[0]

            correct = output_prediction.lower() == target.lower()
            if correct:
                accuracy[example_class]['accurate'] += 1
                if dataset_mode == "cqa" or dataset_mode == "gsm":
                    new_example = cur_base_context + output_numbers + endoftext
                elif dataset_mode == "arithmetic":
                    if few_shot_train:
                        raise NotImplementedError
                    joined_output = "\n".join(output_numbers[:prediction_index + 1])
                    if ENDOFTEXT in joined_output:
                        joined_output = joined_output.split(ENDOFTEXT)[0]
                    new_example = cur_base_context + joined_output + endoftext
                train_examples.append(new_example)
                successful_examples.append(idx)
        except IndexError:
            pass
    return {"successful": successful_examples, "train_examples": train_examples,
            "accuracy": accuracy}


def _get_score(subcounts):
    """官方 get_score：total==0 -> int 0。"""
    if not isinstance(subcounts, dict) or \
            'accurate' not in subcounts or 'total' not in subcounts:
        raise ValueError("subcounts must be a dict with keys accurate/total")
    accurate = subcounts['accurate']
    total = subcounts['total']
    if isinstance(accurate, bool) or not isinstance(accurate, int) or \
            isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("accurate/total must be ints")
    if total == 0:
        return 0
    return accurate / total


def _question_to_context(data_example, dataset_mode, hint=False, direct=False):
    """官方 question_to_context：三数据集上下文组装。"""
    _check_dataset_mode(dataset_mode)
    _check_bool(hint, "hint")
    _check_bool(direct, "direct")
    if dataset_mode == 'cqa':
        if not isinstance(data_example, dict):
            raise ValueError("cqa example must be a dict")
        context = f"Q: {data_example['question']['stem']}\nAnswer Choices:\n"
        for choice in data_example['question']['choices']:
            if hint and (choice['label'].lower() == data_example['answerKey'].lower()):
                context += f"({choice['label'].lower()}) {choice['text']} (CORRECT)\n"
            else:
                context += f"({choice['label'].lower()}) {choice['text']}\n"
        context += "A:"
    elif dataset_mode == 'gsm':
        if not isinstance(data_example, dict):
            raise ValueError("gsm example must be a dict")
        context = f"Q: {data_example['question']}"
        if hint:
            chosen_hint = data_example['answer']
            context += f" ({chosen_hint})"
        context += "\nA:"
    elif dataset_mode == "arithmetic":
        if not isinstance(data_example, str):
            raise ValueError("arithmetic example must be a str")
        context = ""
        splits = data_example.split('Target:')
        for example_split, next_example_split in zip(splits[:-1], splits[1:]):
            if direct and "</scratch>" in example_split:
                context += example_split.split("</scratch>")[-1]
            else:
                context += example_split
            context += "Target:"
            if hint:
                context += " " + next_example_split.split("\n")[-5]
    return context


def _extract_target(data_example, dataset_mode):
    """官方 examples_to_batch 的 target 抽取。"""
    if dataset_mode == "arithmetic":
        if len(data_example.split("\n")) >= 3:
            return data_example.split("\n")[-3]
        return "invalid"
    if dataset_mode == "cqa":
        return data_example['answerKey']
    return data_example['answer'].split("#### ")[-1]


def _eval_examples(dataset, dataset_mode, prompts, hint_prompts, gen_fn, rng,
                   eval_batch_size, accurate_count, train_examples,
                   rationalize=False, p_rationalization=1.0,
                   p_show_hint_save=0.0, direct=False, few_shot_train=False,
                   train_prompts=None):
    """官方 eval_examples + eval_batch/examples_to_batch（去分词）。"""
    _check_str_list(prompts if prompts is not None else [], "prompts")
    _check_str_list(hint_prompts if hint_prompts is not None else [], "hint_prompts")
    _check_bool(rationalize, "rationalize")
    _check_bool(direct, "direct")
    _check_bool(few_shot_train, "few_shot_train")
    if not callable(gen_fn):
        raise ValueError("gen_fn must be callable")
    if not isinstance(rng, random.Random):
        raise ValueError("rng must be a random.Random instance")

    def eval_batch(batch, few_shot_prompts, hint):
        if few_shot_train and dataset_mode == "arithmetic":
            raise NotImplementedError
        cur_few_shot = few_shot_prompts
        answers, classes, base_contexts, input_contexts = [], [], [], []
        for data_class, data_example in batch:
            classes.append(data_class)
            hintless_base_context = _question_to_context(
                data_example, dataset_mode, hint=False, direct=direct)
            base_context = _question_to_context(
                data_example, dataset_mode, hint=hint, direct=direct)
            if dataset_mode == "arithmetic":
                cur_few_shot = base_context.split("\n\n")[:-1]
                base_context = base_context.split("\n\n")[-1]
                hintless_base_context = hintless_base_context.split("\n\n")[-1]
            if rng.random() < p_show_hint_save:
                hintless_base_context = base_context
            if few_shot_train:
                save_context = "\n\n".join(train_prompts) + "\n\n" + hintless_base_context
                base_contexts.append(save_context)
            else:
                base_contexts.append(hintless_base_context)
            if prompts is None:
                context = ""
            else:
                context = "\n\n".join(cur_few_shot) + "\n\n"
            context += base_context
            input_contexts.append(context)
            answers.append(_extract_target(data_example, dataset_mode))
        outputs = gen_fn(input_contexts, hint)
        result = _eval_output(outputs, answers, base_contexts, classes,
                              dataset_mode, accuracy=accurate_count,
                              direct=direct, few_shot_train=few_shot_train)
        train_examples.extend(result["train_examples"])
        return result["successful"]

    main_examples, hint_examples = [], []
    for data_example in dataset:
        main_examples.append(data_example)
        if len(main_examples) == eval_batch_size:
            successful_examples = eval_batch(main_examples, prompts, hint=False)
            for example_idx, example in enumerate(main_examples):
                if (example_idx not in successful_examples) and \
                        (rng.random() < p_rationalization):
                    hint_examples.append(example)
            main_examples = []
        if rationalize and len(hint_examples) >= eval_batch_size:
            cur_hint_examples = hint_examples[:eval_batch_size]
            cur_hint_examples = [
                (hint_example_key + "_r", hint_example)
                for hint_example_key, hint_example in cur_hint_examples
            ]
            eval_batch(cur_hint_examples, hint_prompts, hint=True)
            hint_examples = hint_examples[eval_batch_size:]
    return {'accuracy': accurate_count, 'train_examples': train_examples}


def solve(mode="extract", outputs=None, answers=None, base_contexts=None,
          classes=None, dataset_mode=None, accuracy=None, direct=False,
          few_shot_train=False, endoftext=ENDOFTEXT, counts=None,
          example=None, hint=False, dataset=None, prompts=None,
          hint_prompts=None, gen_fn=None, rng=None, eval_batch_size=None,
          rationalize=False, p_rationalization=1.0, p_show_hint_save=0.0,
          train_prompts=None):
    """统一入口（判题协议）：按 mode 分发到 extract / score / context / eval。"""
    if mode == "score":
        return _get_score(counts)
    if mode == "context":
        if dataset_mode is None:
            raise ValueError("dataset_mode is required")
        return _question_to_context(example, dataset_mode, hint=hint, direct=direct)
    if mode == "extract":
        if dataset_mode is None:
            raise ValueError("dataset_mode is required")
        return _eval_output(outputs, answers, base_contexts, classes, dataset_mode,
                            accuracy=accuracy, direct=direct,
                            few_shot_train=few_shot_train, endoftext=endoftext)
    if mode == "eval":
        if dataset_mode is None:
            raise ValueError("dataset_mode is required")
        if not isinstance(dataset, (list, tuple)):
            raise ValueError("dataset must be a list of (key, example) pairs")
        for pair in dataset:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                    and isinstance(pair[0], str)):
                raise ValueError("each dataset item must be a (key, example) pair")
        if not isinstance(eval_batch_size, int) or isinstance(eval_batch_size, bool) \
                or eval_batch_size < 1:
            raise ValueError("eval_batch_size must be a positive int")
        for name, value in (("p_rationalization", p_rationalization),
                            ("p_show_hint_save", p_show_hint_save)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1]")
        if train_prompts is None:
            train_prompts = []
        return _eval_examples(list(dataset), dataset_mode, prompts, hint_prompts,
                              gen_fn, rng, eval_batch_size, {}, [],
                              rationalize=rationalize,
                              p_rationalization=p_rationalization,
                              p_show_hint_save=p_show_hint_save, direct=direct,
                              few_shot_train=few_shot_train,
                              train_prompts=list(train_prompts))
    raise ValueError(f"Unsupported mode {mode!r}")
